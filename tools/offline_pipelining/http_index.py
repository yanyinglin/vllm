#!/usr/bin/env python3
"""
Single Stage Launcher for ZeroMQ Pipeline Parallelism

This script launches a single pipeline stage using external pipeline mode.
Each stage should be launched independently.
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
from typing import Optional

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


class StageLauncher:
    """Launches a single pipeline stage"""
    
    def __init__(
        self,
        pipeline_dir: str,
        stage_idx: int,
        num_stages: int,
        base_port: int = 15550,
        api_port: int = 5000,
        gpu_id: int = 0,
        vllm_source_dir: Optional[str] = None,
        prev_stage_service_name: Optional[str] = None,
    ):
        self.pipeline_dir = Path(pipeline_dir)
        self.stage_idx = stage_idx
        self.num_stages = num_stages
        self.base_port = base_port
        self.api_port = api_port
        self.gpu_id = gpu_id
        self.vllm_source_dir = vllm_source_dir
        self.local_ip = get_ip()
        self.prev_stage_service_name = prev_stage_service_name
        self.process: Optional[subprocess.Popen] = None
        self.log_dir = Path("./zeromq_pp_logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Validate pipeline directory
        stage_dir = self.pipeline_dir / f"stage_{stage_idx}"
        if not stage_dir.exists():
            raise ValueError(f"Stage directory not found: {stage_dir}")
    
    def _get_vllm_cmd(self) -> str:
        """Get vLLM command based on source mode"""
        if self.vllm_source_dir:
            if not Path(self.vllm_source_dir).exists():
                raise ValueError(f"VLLM_SOURCE_DIR does not exist: {self.vllm_source_dir}")
            return "python -m vllm.entrypoints.cli.main serve"
        return "vllm serve"
    
    def _build_stage_command(self) -> list[str]:
        """Build command for this pipeline stage using bind mode architecture.
        
        Bind Mode Architecture:
        - PUSH sockets bind to ports (senders bind, receivers connect)
        - PULL sockets connect to Service addresses (receivers connect to senders)
        - This allows Kubernetes Service load balancing
        
        Communication Flow:
        - Forward path: Stage i binds PUSH socket, Stage i+1 connects PULL socket to Stage i
        - Return path: Last stage binds PUSH socket, Stage 0 connects PULL socket to last stage
        
        Each stage only needs to know:
        - Its own port (for binding PUSH socket if not last stage)
        - Previous stage's address (for connecting PULL socket if not first stage)
          Uses `--pipeline-prev-stage-service-addr` (clear name: previous stage's PUSH Service address)
        """
        is_first = (self.stage_idx == 0)
        is_last = (self.stage_idx == self.num_stages - 1)
        
        stage_dir = self.pipeline_dir / f"stage_{self.stage_idx}"
        
        cmd = self._get_vllm_cmd().split()
        cmd.append(str(stage_dir))
        cmd.extend([
            "--pipeline-stage-mode", "external",
            "--pipeline-stage-idx", str(self.stage_idx),
            "--pipeline-total-stages", str(self.num_stages),
            "--tensor-parallel-size", "1",
        ])
        
        # Forward path: Stage i binds PUSH socket, Stage i+1 connects PULL socket to Stage i
        if not is_last:
            # Non-last stages: bind PUSH socket to local port
            # All stages use the same base_port (each pod has its own network namespace in Kubernetes)
            bind_port = self.base_port
            cmd.extend([
                "--pipeline-local-bind-port",
                str(bind_port),
            ])
        
        if not is_first:
            # Non-first stages: connect PULL socket to previous stage's PUSH Service
            # In bind mode: previous stage binds PUSH, this stage connects PULL to it
            # Previous stage binds to base_port (all stages use the same port)
            prev_bind_port = self.base_port
            if self.prev_stage_service_name:
                # Use Kubernetes service name if provided
                prev_addr = f"{self.prev_stage_service_name}:{prev_bind_port}"
            else:
                # Fallback to IP address
                prev_addr = f"{self.local_ip}:{prev_bind_port}"
            cmd.extend([
                "--pipeline-prev-stage-service-addr",  # Clear name: previous stage's PUSH Service address
                prev_addr,
            ])
        
        # Stage 0 exposes HTTP API; others are external PP workers
        if is_first:
            cmd.extend(["--port", str(self.api_port)])
        else:
            cmd.append("--external-pp-worker")
        
        return cmd
    
    def launch(self):
        """Launch this pipeline stage"""
        is_first = (self.stage_idx == 0)
        is_last = (self.stage_idx == self.num_stages - 1)
        
        cmd = self._build_stage_command()
        log_file = self.log_dir / f"stage_{self.stage_idx}.log"
        
        print("=" * 50)
        print(f"Launching Stage {self.stage_idx}")
        print("=" * 50)
        print(f"Pipeline Directory: {self.pipeline_dir}")
        print(f"Stage Index: {self.stage_idx}")
        print(f"Total Stages: {self.num_stages}")
        print(f"Local IP: {self.local_ip}")
        print(f"Base ZeroMQ Port: {self.base_port}")
        print(f"GPU ID: {self.gpu_id}")
        if self.vllm_source_dir:
            print(f"vLLM Source Directory: {self.vllm_source_dir}")
        print(f"Log File: {log_file}")
        print()
        
        print(f"Stage {self.stage_idx} Configuration:")
        print(f"  Model: {self.pipeline_dir / f'stage_{self.stage_idx}'}")
        print(f"  GPU: {self.gpu_id}")
        
        if not is_last:
            bind_port = self.base_port
            print(f"  Forward Bind Port: {bind_port} (PUSH socket binds, Stage {self.stage_idx + 1} will connect PULL to this)")
        if not is_first:
            prev_bind_port = self.base_port
            print(f"  Previous Stage Address: {self.local_ip}:{prev_bind_port} (PULL socket connects to Stage {self.stage_idx - 1}'s PUSH)")
            print(f"    Using --pipeline-prev-stage-service-addr (clear name for previous stage's PUSH Service)")
        if is_first:
            print(f"  API Port: {self.api_port}")
        else:
            print(f"  Mode: external-pp-worker (no HTTP API)")
        print(f"  Command: {' '.join(cmd)}")
        print()
        
        # Set environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        if self.vllm_source_dir:
            pythonpath = env.get("PYTHONPATH", "")
            if pythonpath:
                env["PYTHONPATH"] = f"{self.vllm_source_dir}:{pythonpath}"
            else:
                env["PYTHONPATH"] = self.vllm_source_dir
        
        # Launch process
        with open(log_file, "w") as f:
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
            )
        
        print(f"Stage {self.stage_idx} launched with PID: {self.process.pid}")
        print(f"Log file: {log_file}")
        print()
        
        # Check if process is still running
        time.sleep(3)
        if self.process.poll() is not None:
            print(f"ERROR: Stage {self.stage_idx} (PID: {self.process.pid}) failed to start!")
            print(f"Check log file: {log_file}")
            with open(log_file, "r") as f:
                lines = f.readlines()
                print("Last 20 lines of log:")
                for line in lines[-20:]:
                    print(f"  {line.rstrip()}")
            raise RuntimeError(f"Stage {self.stage_idx} failed to start")
        
        print(f"Stage {self.stage_idx} is running")
        print("=" * 50)
        print()
    
    def cleanup(self):
        """Cleanup this pipeline stage"""
        if self.process is None:
            return
        
        print("\nCleaning up stage...")
        if self.process.poll() is None:
            print(f"Killing process {self.process.pid} (stage {self.stage_idx})")
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            except Exception as e:
                print(f"Error killing process {self.process.pid}: {e}")
        print("Cleanup complete")


def main():
    parser = argparse.ArgumentParser(
        description="Single Stage Launcher for ZeroMQ Pipeline Parallelism"
    )
    parser.add_argument(
        "--pipeline-dir",
        type=str,
        default=None,
        help="Directory containing pipeline stages",
    )
    parser.add_argument(
        "--stage-idx",
        type=int,
        default=None,
        help="Stage index to launch (0-based)",
    )
    parser.add_argument(
        "--num-stages",
        type=int,
        default=None,
        help="Total number of pipeline stages",
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
        "--gpu-id",
        type=int,
        default=None,
        help="GPU ID to use (default: 0)",
    )
    parser.add_argument(
        "--vllm-source-dir",
        type=str,
        default=None,
        help="vLLM source directory (optional, for source mode)",
    )
    parser.add_argument(
        "--prev-stage-service-name",
        type=str,
        default=None,
        help="Previous stage Kubernetes service name (optional)",
    )
    
    args = parser.parse_args()
    
    # Read from environment variables if not provided via command line
    pipeline_dir = args.pipeline_dir or os.environ.get("PIPELINE_DIR")
    stage_idx = args.stage_idx
    if stage_idx is None:
        stage_idx_str = os.environ.get("PIPELINE_STAGE_IDX")
        if stage_idx_str:
            stage_idx = int(stage_idx_str)
    
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
    
    gpu_id = args.gpu_id
    if gpu_id is None:
        gpu_id_str = os.environ.get("GPU_IDS")
        if gpu_id_str:
            # GPU_IDS can be comma-separated, but for single stage we use the first one
            gpu_ids = [int(x.strip()) for x in gpu_id_str.split(",")]
            gpu_id = gpu_ids[0] if gpu_ids else 0
        else:
            gpu_id = 0
    
    vllm_source_dir = args.vllm_source_dir or os.environ.get("VLLM_SOURCE_DIR")
    prev_stage_service_name = args.prev_stage_service_name or os.environ.get("PREV_STAGE_SERVICE_NAME")
    
    # Validate required arguments
    if pipeline_dir is None:
        raise ValueError("--pipeline-dir or PIPELINE_DIR environment variable is required")
    if stage_idx is None:
        raise ValueError("--stage-idx or PIPELINE_STAGE_IDX environment variable is required")
    if num_stages is None:
        raise ValueError("--num-stages or NUM_STAGES environment variable is required")
    
    # Validate stage index
    if stage_idx < 0 or stage_idx >= num_stages:
        raise ValueError(f"Invalid stage index: {stage_idx} (must be 0-{num_stages-1})")
    
    # Create stage launcher
    launcher = StageLauncher(
        pipeline_dir=pipeline_dir,
        stage_idx=stage_idx,
        num_stages=num_stages,
        base_port=base_port,
        api_port=api_port,
        gpu_id=gpu_id,
        vllm_source_dir=vllm_source_dir,
        prev_stage_service_name=prev_stage_service_name,
    )
    
    # Register cleanup handler
    atexit.register(launcher.cleanup)
    
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
        launcher.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Launch stage
    try:
        launcher.launch()
        
        print(f"Stage {stage_idx} worker running")
        print("Press Ctrl+C to stop")
        print()
        
        # Keep process alive
        try:
            while True:
                time.sleep(1)
                # Check if process is still running
                if launcher.process and launcher.process.poll() is not None:
                    print(f"Stage {stage_idx} process exited")
                    break
        except KeyboardInterrupt:
            pass
        
    except KeyboardInterrupt:
        print("\nShutting down...")
        launcher.cleanup()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        launcher.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()
