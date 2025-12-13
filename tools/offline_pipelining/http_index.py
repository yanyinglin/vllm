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
        api_port: int = 5000,
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
        """Build command for a pipeline stage using bind mode architecture.
        
        Each stage only needs to know:
        - Its own IP and port (for binding)
        - Previous stage's IP and port (for connecting PULL socket)
        - Last stage's IP and port (Stage 0 for return path)
        
        Bind mode:
        - PUSH sockets bind to ports (senders)
        - PULL sockets connect to Service addresses (receivers)
        - This allows Kubernetes Service load balancing
        
        Forward path: stage_i binds PUSH, stage_{i+1} connects PULL
        Return path: last stage binds PUSH, stage 0 connects PULL
        """
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
        
        # Forward path: stage_i binds PUSH, stage_{i+1} connects PULL
        if not is_last:
            # Non-last stages: bind PUSH socket to local port on this stage's IP
            bind_port = self.base_port + stage_idx
            cmd.extend([
                "--pipeline-local-bind-port",
                str(bind_port),
            ])
        
        if not is_first:
            # Non-first stages: connect PULL socket to previous stage's PUSH Service
            prev_bind_port = self.base_port + stage_idx - 1
            cmd.extend([
                "--pipeline-next-stage-addr",
                f"{self.local_ip}:{prev_bind_port}",
            ])
        
        # Return path: last stage binds PUSH, stage 0 connects PULL
        if is_last and not is_first:
            # Last stage: bind PUSH socket for return path on this stage's IP
            return_bind_port = self.base_port + self.num_stages
            cmd.extend([
                "--pipeline-local-bind-port",
                str(return_bind_port),
            ])
        
        if is_first and not is_last:
            # Stage 0: connect PULL socket to last stage's return Service
            return_bind_port = self.base_port + self.num_stages
            cmd.extend([
                "--pipeline-prev-stage-addr",
                f"{self.local_ip}:{return_bind_port}",
            ])
        
        # Stage 0 exposes HTTP API; others are external PP workers
        if is_first:
            cmd.extend(["--port", str(self.api_port)])
        else:
            cmd.append("--external-pp-worker")
        
        return cmd
    
    def launch_stages(self, single_stage_idx: Optional[int] = None):
        """Launch all pipeline stages or a single stage"""
        print("=" * 50)
        print("ZeroMQ Pipeline Parallelism HTTP Entry Point")
        print("=" * 50)
        print(f"Pipeline Directory: {self.pipeline_dir}")
        print(f"Number of Stages: {self.num_stages}")
        if single_stage_idx is not None:
            print(f"Single Stage Mode: Launching only stage {single_stage_idx}")
        print(f"Local IP: {self.local_ip}")
        print(f"Base ZeroMQ Port: {self.base_port}")
        print(f"API Port: {self.api_port}")
        print(f"GPU IDs: {self.gpu_ids}")
        if self.vllm_source_dir:
            print(f"vLLM Source Directory: {self.vllm_source_dir}")
        print("=" * 50)
        print()
        
        if single_stage_idx is not None:
            if single_stage_idx < 0 or single_stage_idx >= self.num_stages:
                raise ValueError(f"Invalid stage index: {single_stage_idx} (must be 0-{self.num_stages-1})")
            self._launch_stage(single_stage_idx)
        else:
            for stage_idx in range(self.num_stages):
                self._launch_stage(stage_idx)
                time.sleep(3)  # Wait between launches
        
        print()
        print("=" * 50)
        if single_stage_idx is not None:
            print(f"Stage {single_stage_idx} launched successfully!")
        else:
            print("All stages launched successfully!")
        print("=" * 50)
        print(f"Stage PIDs: {self.stage_pids}")
        print(f"Logs directory: {self.log_dir}")
        if single_stage_idx is None or single_stage_idx == 0:
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
        
        if not is_last:
            bind_port = self.base_port + stage_idx
            print(f"  Forward Bind Port: {bind_port} (PUSH socket binds, allows next stage to connect)")
        if not is_first:
            prev_bind_port = self.base_port + stage_idx - 1
            print(f"  Previous Stage Service: {self.local_ip}:{prev_bind_port} (PULL socket connects)")
        if is_last and not is_first:
            return_bind_port = self.base_port + self.num_stages
            print(f"  Return Bind Port: {return_bind_port} (PUSH socket binds for return path)")
        if is_first and not is_last:
            return_bind_port = self.base_port + self.num_stages
            print(f"  Return Service: {self.local_ip}:{return_bind_port} (PULL socket connects to last stage)")
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
        default=None,
        help="Directory containing pipeline stages",
    )
    parser.add_argument(
        "--num-stages",
        type=int,
        default=None,
        help="Number of pipeline stages",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=None,
        help="Starting ZeroMQ port (default: 15550)",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=None,
        help="HTTP API port for stage 0 (default: 5000)",
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
        default=None,
        help="Host to bind HTTP server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port for HTTP entry point server (default: 8080)",
    )
    parser.add_argument(
        "--vllm-source-dir",
        type=str,
        default=None,
        help="vLLM source directory (optional, for source mode)",
    )
    parser.add_argument(
        "--single-stage-idx",
        type=int,
        default=None,
        help="Launch only a single stage (for distributed deployment)",
    )
    
    args = parser.parse_args()
    
    # Read from environment variables if not provided via command line
    pipeline_dir = args.pipeline_dir or os.environ.get("PIPELINE_DIR")
    num_stages = args.num_stages
    if num_stages is None:
        num_stages_str = os.environ.get("NUM_STAGES")
        if num_stages_str:
            num_stages = int(num_stages_str)
    
    base_port = args.base_port
    if base_port is None:
        base_port_str = os.environ.get("BASE_PORT", "15550")
        base_port = int(base_port_str)
    
    api_port = args.api_port
    if api_port is None:
        api_port_str = os.environ.get("API_PORT", "5000")
        api_port = int(api_port_str)
    
    gpu_ids_str = args.gpu_ids or os.environ.get("GPU_IDS")
    host = args.host or os.environ.get("HOST", "127.0.0.1")
    port = args.port
    if port is None:
        port_str = os.environ.get("PORT", "8080")
        port = int(port_str)
    
    single_stage_idx = args.single_stage_idx
    if single_stage_idx is None:
        single_stage_idx_str = os.environ.get("PIPELINE_STAGE_IDX")
        if single_stage_idx_str:
            single_stage_idx = int(single_stage_idx_str)
    
    # Validate required arguments
    if pipeline_dir is None:
        raise ValueError("--pipeline-dir or PIPELINE_DIR environment variable is required")
    if num_stages is None:
        raise ValueError("--num-stages or NUM_STAGES environment variable is required")
    
    # Parse GPU IDs
    if gpu_ids_str:
        gpu_ids = [int(x.strip()) for x in gpu_ids_str.split(",")]
    else:
        gpu_ids = list(range(num_stages))
    
    # Check VLLM_SOURCE_DIR environment variable
    vllm_source_dir = args.vllm_source_dir or os.environ.get("VLLM_SOURCE_DIR")
    
    # Create pipeline manager
    pipeline_manager = PipelineManager(
        pipeline_dir=pipeline_dir,
        num_stages=num_stages,
        base_port=base_port,
        api_port=api_port,
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
        pipeline_manager.launch_stages(single_stage_idx=single_stage_idx)
        
        # Wait for stage 0 to be ready (only if stage 0 is launched)
        if single_stage_idx is None or single_stage_idx == 0:
            pipeline_manager.wait_for_stage0_ready()
        
        # Create and run HTTP server (only for stage 0 or all stages mode)
        if single_stage_idx is None or single_stage_idx == 0:
            app = create_app(pipeline_manager)
            
            print("=" * 50)
            print("HTTP Entry Point Server Starting")
            print("=" * 50)
            print(f"Server URL: http://{host}:{port}")
            print(f"Stage 0 API: http://localhost:{api_port}")
            print("=" * 50)
            print()
            print("Press Ctrl+C to stop all stages")
            print()
            
            uvicorn.run(app, host=host, port=port, log_level="info")
        else:
            # For non-stage-0 workers, just keep the process running
            print("=" * 50)
            print(f"Stage {single_stage_idx} worker running")
            print("=" * 50)
            print("Press Ctrl+C to stop")
            print()
            
            # Keep process alive
            try:
                while True:
                    time.sleep(1)
                    # Check if process is still running
                    if pipeline_manager.processes and pipeline_manager.processes[0].poll() is not None:
                        print(f"Stage {single_stage_idx} process exited")
                        break
            except KeyboardInterrupt:
                pass
        
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
