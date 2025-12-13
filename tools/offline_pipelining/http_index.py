#!/usr/bin/env python3
"""
HTTP Entry Point for ZeroMQ Pipeline Parallelism

This script launches all pipeline stages using external pipeline mode and
exposes an HTTP server that forwards inference requests to stage 0.
"""
import argparse
import atexit
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

try:
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import StreamingResponse
    import httpx
    import uvicorn
except ImportError:
    print("Error: Required packages not installed. Please install: fastapi, httpx, uvicorn")
    sys.exit(1)

try:
    from vllm.utils.network_utils import get_ip
except ImportError:
    # Fallback if vLLM is not in path
    def get_ip() -> str:
        """Get local IP address"""
        host_ip = os.environ.get("VLLM_HOST_IP")
        if host_ip:
            return host_ip
        
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            pass
        
        return "127.0.0.1"


class PipelineManager:
    """Manages pipeline stage processes"""
    
    def __init__(
        self,
        pipeline_dir: str,
        num_stages: int,
        base_port: int = 15550,
        api_port: int = 8000,
        gpu_ids: List[int] = None,
        vllm_source_dir: Optional[str] = None,
    ):
        self.pipeline_dir = Path(pipeline_dir)
        self.num_stages = num_stages
        self.base_port = base_port
        self.api_port = api_port
        self.gpu_ids = gpu_ids or list(range(num_stages))
        self.vllm_source_dir = vllm_source_dir
        self.local_ip = get_ip()
        self.processes: List[subprocess.Popen] = []
        self.stage_pids: List[int] = []
        self.log_dir = Path("./zeromq_pp_logs")
        self.log_dir.mkdir(exist_ok=True)
        
        if len(self.gpu_ids) < num_stages:
            raise ValueError(
                f"GPU_IDS has only {len(self.gpu_ids)} entries, "
                f"but NUM_STAGES={num_stages}"
            )
        
        # Validate pipeline directory
        for i in range(num_stages):
            stage_dir = self.pipeline_dir / f"stage_{i}"
            if not stage_dir.exists():
                raise ValueError(f"Stage directory not found: {stage_dir}")
    
    def _get_vllm_cmd(self) -> str:
        """Get vLLM command based on source mode"""
        if self.vllm_source_dir:
            if not Path(self.vllm_source_dir).exists():
                raise ValueError(f"VLLM_SOURCE_DIR does not exist: {self.vllm_source_dir}")
            return "python -m vllm.entrypoints.cli.main serve"
        return "vllm serve"
    
    def _build_stage_command(self, stage_idx: int) -> List[str]:
        """Build command for a pipeline stage"""
        is_first = (stage_idx == 0)
        is_last = (stage_idx == self.num_stages - 1)
        
        stage_dir = self.pipeline_dir / f"stage_{stage_idx}"
        
        cmd = self._get_vllm_cmd().split()
        cmd.append(str(stage_dir))
        cmd.extend([
            "--pipeline-stage-mode", "external",
            "--pipeline-stage-idx", str(stage_idx),
            "--pipeline-total-stages", str(self.num_stages),
            "--tensor-parallel-size", "1",
        ])
        
        # Set next stage address (except for last stage)
        if not is_last:
            next_port = self.base_port + stage_idx
            cmd.extend([
                "--pipeline-next-stage-addr",
                f"{self.local_ip}:{next_port}",
            ])
        
        # Set local listen port
        if not is_first:
            # Non-first stages: listen on forward path port
            listen_port = self.base_port + stage_idx - 1
            cmd.extend([
                "--pipeline-local-listen-port",
                str(listen_port),
            ])
        elif is_first and not is_last:
            # Stage 0: listen on return port
            return_port = self.base_port + self.num_stages
            cmd.extend([
                "--pipeline-local-listen-port",
                str(return_port),
            ])
        
        # Last stage needs prev_stage_addr for return path
        if is_last:
            return_port = self.base_port + self.num_stages
            cmd.extend([
                "--pipeline-prev-stage-addr",
                f"{self.local_ip}:{return_port}",
            ])
        
        # Stage 0 exposes HTTP API; others are external PP workers
        if is_first:
            cmd.extend(["--port", str(self.api_port)])
        else:
            cmd.append("--external-pp-worker")
        
        return cmd
    
    def launch_stages(self):
        """Launch all pipeline stages"""
        print("=" * 50)
        print("ZeroMQ Pipeline Parallelism HTTP Entry Point")
        print("=" * 50)
        print(f"Pipeline Directory: {self.pipeline_dir}")
        print(f"Number of Stages: {self.num_stages}")
        print(f"Local IP: {self.local_ip}")
        print(f"Base ZeroMQ Port: {self.base_port}")
        print(f"API Port: {self.api_port}")
        print(f"GPU IDs: {self.gpu_ids}")
        if self.vllm_source_dir:
            print(f"vLLM Source Directory: {self.vllm_source_dir}")
        print("=" * 50)
        print()
        
        for stage_idx in range(self.num_stages):
            self._launch_stage(stage_idx)
            time.sleep(3)  # Wait between launches
        
        print()
        print("=" * 50)
        print("All stages launched successfully!")
        print("=" * 50)
        print(f"Stage PIDs: {self.stage_pids}")
        print(f"Logs directory: {self.log_dir}")
        print(f"API endpoint: http://localhost:{self.api_port}")
        print()
    
    def _launch_stage(self, stage_idx: int):
        """Launch a single pipeline stage"""
        is_first = (stage_idx == 0)
        is_last = (stage_idx == self.num_stages - 1)
        
        cmd = self._build_stage_command(stage_idx)
        log_file = self.log_dir / f"stage_{stage_idx}.log"
        gpu_id = self.gpu_ids[stage_idx]
        
        print(f"Launching Stage {stage_idx}:")
        print(f"  Model: {self.pipeline_dir / f'stage_{stage_idx}'}")
        print(f"  Log: {log_file}")
        print(f"  GPU: {gpu_id}")
        
        if not is_first:
            listen_port = self.base_port + stage_idx - 1
            print(f"  Forward Listen Port: {listen_port} (receives from stage {stage_idx - 1})")
        if is_first and not is_last:
            return_port = self.base_port + self.num_stages
            print(f"  Return Listen Port: {return_port} (receives final results from last stage)")
        if not is_last:
            next_port = self.base_port + stage_idx
            print(f"  Next Stage: {self.local_ip}:{next_port}")
        if is_last:
            return_port = self.base_port + self.num_stages
            print(f"  Return Address: {self.local_ip}:{return_port} (sends final results to stage 0)")
        if is_first:
            print(f"  API Port: {self.api_port}")
        else:
            print(f"  Mode: external-pp-worker (no HTTP API)")
        print(f"  Command: {' '.join(cmd)}")
        print()
        
        # Set environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        if self.vllm_source_dir:
            pythonpath = env.get("PYTHONPATH", "")
            if pythonpath:
                env["PYTHONPATH"] = f"{self.vllm_source_dir}:{pythonpath}"
            else:
                env["PYTHONPATH"] = self.vllm_source_dir
        
        # Launch process
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
            )
        
        self.processes.append(process)
        self.stage_pids.append(process.pid)
        
        print(f"Stage {stage_idx} launched with PID: {process.pid}")
        
        # Check if process is still running
        time.sleep(3)
        if process.poll() is not None:
            print(f"ERROR: Stage {stage_idx} (PID: {process.pid}) failed to start!")
            print(f"Check log file: {log_file}")
            with open(log_file, "r") as f:
                lines = f.readlines()
                print("Last 20 lines of log:")
                for line in lines[-20:]:
                    print(f"  {line.rstrip()}")
            raise RuntimeError(f"Stage {stage_idx} failed to start")
        
        print(f"Stage {stage_idx} is running")
        print()
    
    def wait_for_stage0_ready(self, max_retries: int = 30, retry_interval: int = 2):
        """Wait for stage 0 API server to be ready"""
        print("Waiting for API server to be ready...")
        url = f"http://localhost:{self.api_port}/health"
        
        for i in range(max_retries):
            try:
                response = httpx.get(url, timeout=5.0)
                if response.status_code == 200:
                    print("API server is ready")
                    return True
            except Exception:
                pass
            
            if i < max_retries - 1:
                print(f"  Waiting for API server... (attempt {i + 1}/{max_retries})")
                time.sleep(retry_interval)
        
        print(f"Warning: API server health check failed after {max_retries} attempts")
        return False
    
    def cleanup(self):
        """Cleanup all pipeline stages"""
        print("\nCleaning up pipeline stages...")
        for i, process in enumerate(self.processes):
            if process.poll() is None:
                print(f"Killing process {process.pid} (stage {i})")
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                except Exception as e:
                    print(f"Error killing process {process.pid}: {e}")
        print("Cleanup complete")


def create_app(pipeline_manager: PipelineManager) -> FastAPI:
    """Create FastAPI app with proxy endpoints"""
    app = FastAPI(title="ZeroMQ Pipeline HTTP Entry Point")
    stage0_url = f"http://localhost:{pipeline_manager.api_port}"
    
    @app.get("/health")
    async def health():
        """Health check endpoint"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{stage0_url}/health", timeout=5.0)
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                )
        except Exception as e:
            return Response(
                content=f'{{"error": "Health check failed: {str(e)}"}}',
                status_code=503,
                media_type="application/json",
            )
    
    @app.get("/pipeline/status")
    async def pipeline_status():
        """Get pipeline status"""
        status = {
            "num_stages": pipeline_manager.num_stages,
            "stage_pids": pipeline_manager.stage_pids,
            "api_port": pipeline_manager.api_port,
            "base_port": pipeline_manager.base_port,
            "local_ip": pipeline_manager.local_ip,
        }
        
        # Check if processes are still running
        process_status = []
        for i, process in enumerate(pipeline_manager.processes):
            is_running = process.poll() is None
            process_status.append({
                "stage_idx": i,
                "pid": process.pid,
                "running": is_running,
            })
        status["stages"] = process_status
        
        return status
    
    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
    async def proxy_request(request: Request, path: str):
        """Proxy all requests to stage 0"""
        url = f"{stage0_url}/{path}"
        
        # Get request body
        body = await request.body()
        
        # Get query parameters
        query_params = dict(request.query_params)
        
        # Forward request
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.request(
                    method=request.method,
                    url=url,
                    content=body,
                    params=query_params,
                    headers={k: v for k, v in request.headers.items() 
                            if k.lower() not in ["host", "content-length"]},
                )
                
                # Handle streaming responses
                if "text/event-stream" in response.headers.get("content-type", ""):
                    async def stream_generator():
                        async for chunk in response.aiter_bytes():
                            yield chunk
                    
                    return StreamingResponse(
                        stream_generator(),
                        status_code=response.status_code,
                        headers=dict(response.headers),
                        media_type=response.headers.get("content-type"),
                    )
                
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.headers.get("content-type"),
                )
        except Exception as e:
            return Response(
                content=f'{{"error": "Proxy request failed: {str(e)}"}}',
                status_code=502,
                media_type="application/json",
            )
    
    return app


def main():
    parser = argparse.ArgumentParser(
        description="HTTP Entry Point for ZeroMQ Pipeline Parallelism"
    )
    parser.add_argument(
        "--pipeline-dir",
        type=str,
        required=True,
        help="Directory containing pipeline stages",
    )
    parser.add_argument(
        "--num-stages",
        type=int,
        required=True,
        help="Number of pipeline stages",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=15550,
        help="Starting ZeroMQ port (default: 15550)",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="HTTP API port for stage 0 (default: 8000)",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated GPU IDs (default: 0,1,2,3,...)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind HTTP server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for HTTP entry point server (default: 8080)",
    )
    parser.add_argument(
        "--vllm-source-dir",
        type=str,
        default=None,
        help="vLLM source directory (optional, for source mode)",
    )
    
    args = parser.parse_args()
    
    # Parse GPU IDs
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
    else:
        gpu_ids = list(range(args.num_stages))
    
    # Check VLLM_SOURCE_DIR environment variable
    vllm_source_dir = args.vllm_source_dir or os.environ.get("VLLM_SOURCE_DIR")
    
    # Create pipeline manager
    pipeline_manager = PipelineManager(
        pipeline_dir=args.pipeline_dir,
        num_stages=args.num_stages,
        base_port=args.base_port,
        api_port=args.api_port,
        gpu_ids=gpu_ids,
        vllm_source_dir=vllm_source_dir,
    )
    
    # Register cleanup handler
    atexit.register(pipeline_manager.cleanup)
    
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
        pipeline_manager.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Launch pipeline stages
    try:
        pipeline_manager.launch_stages()
        
        # Wait for stage 0 to be ready
        pipeline_manager.wait_for_stage0_ready()
        
        # Create and run HTTP server
        app = create_app(pipeline_manager)
        
        print("=" * 50)
        print("HTTP Entry Point Server Starting")
        print("=" * 50)
        print(f"Server URL: http://{args.host}:{args.port}")
        print(f"Stage 0 API: http://localhost:{args.api_port}")
        print("=" * 50)
        print()
        print("Press Ctrl+C to stop all stages")
        print()
        
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        
    except KeyboardInterrupt:
        print("\nShutting down...")
        pipeline_manager.cleanup()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        pipeline_manager.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()


