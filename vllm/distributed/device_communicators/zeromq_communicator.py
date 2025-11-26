# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
ZeroMQ communication layer for pipeline parallelism in external stage mode.

This module provides ZeroMQ-based communication for pipeline stages that run
as independent vLLM instances, allowing them to communicate across different
machines via TCP/IP.
"""

import io
import pickle
from typing import Any

import torch
import zmq

from vllm.distributed.parallel_state import TensorMetadata, _split_tensor_dict
from vllm.logger import init_logger

logger = init_logger(__name__)


class ZeroMQPPCommunicator:
    """ZeroMQ communicator for pipeline parallelism in external stage mode.
    
    This communicator supports connecting to external stages via IP:Port,
    allowing each pipeline stage to run as an independent vLLM instance.
    """

    def __init__(
        self,
        stage_idx: int,
        num_stages: int,
        local_listen_port: int | None = None,
        next_stage_addr: str | None = None,  # "ip:port" format
        prev_stage_addr: str | None = None,  # "ip:port" format
        device: str = "cuda",
        timeout_ms: int = 30000,
    ):
        """Initialize ZeroMQ communicator.
        
        Args:
            stage_idx: Current stage index (0-based)
            num_stages: Total number of stages
            local_listen_port: Local port to bind for receiving data from previous stage
            next_stage_addr: Address of next stage in "ip:port" format
            prev_stage_addr: Address of previous stage in "ip:port" format (optional)
            device: Target device (e.g., "cuda", "cpu")
            timeout_ms: Socket timeout in milliseconds
        """
        self.stage_idx = stage_idx
        self.num_stages = num_stages
        self.device = device
        self.timeout_ms = timeout_ms
        
        self.is_first = stage_idx == 0
        self.is_last = stage_idx == num_stages - 1
        
        # Create ZeroMQ context
        self.context = zmq.Context()
        
        # Sockets for forward communication (current stage -> next stage)
        self.push_socket: zmq.Socket | None = None
        self.pull_socket: zmq.Socket | None = None
        
        self._setup_sockets(local_listen_port, next_stage_addr, prev_stage_addr)
    
    def _setup_sockets(
        self,
        local_listen_port: int | None,
        next_stage_addr: str | None,
        prev_stage_addr: str | None,
    ) -> None:
        """Setup ZeroMQ sockets for communication."""
        # Forward direction: send to next stage
        if not self.is_last:
            if next_stage_addr is None:
                raise ValueError(
                    f"Stage {self.stage_idx}: next_stage_addr must be provided for non-last stages"
                )
            
            self.push_socket = self.context.socket(zmq.PUSH)
            # Connect to next stage's PULL socket
            self.push_socket.connect(f"tcp://{next_stage_addr}")
            logger.info(
                f"Stage {self.stage_idx} connected PUSH socket to {next_stage_addr}"
            )
        
        # Backward direction: receive from previous stage
        if not self.is_first:
            if local_listen_port is None:
                raise ValueError(
                    f"Stage {self.stage_idx}: local_listen_port must be provided for non-first stages"
                )
            
            self.pull_socket = self.context.socket(zmq.PULL)
            # Bind to local port for receiving
            self.pull_socket.bind(f"tcp://*:{local_listen_port}")
            self.pull_socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            logger.info(
                f"Stage {self.stage_idx} bound PULL socket to port {local_listen_port}"
            )
    
    def send_tensor_dict(
        self,
        tensor_dict: dict[str, torch.Tensor | Any],
        dst: int | None = None,
    ) -> None:
        """Send tensor dictionary to next stage.
        
        Args:
            tensor_dict: Dictionary containing tensors and other objects
            dst: Destination stage index (ignored, always sends to next stage)
        """
        if self.is_last:
            raise RuntimeError("Last stage cannot send tensor dict")
        
        if self.push_socket is None:
            raise RuntimeError("Push socket not initialized")
        
        # Split tensor dict into metadata and tensors
        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        
        # Serialize metadata
        metadata_bytes = pickle.dumps(metadata_list)
        
        # Serialize all tensors
        tensor_bytes_list = []
        for tensor in tensor_list:
            # Move tensor to CPU for serialization
            buffer = io.BytesIO()
            torch.save(tensor.cpu(), buffer)
            tensor_bytes_list.append(buffer.getvalue())
        
        # Send metadata: size, data, num_tensors
        self.push_socket.send_multipart([
            len(metadata_bytes).to_bytes(8, 'big'),  # metadata size (8 bytes)
            metadata_bytes,  # metadata
            len(tensor_bytes_list).to_bytes(8, 'big'),  # number of tensors
        ])
        
        # Send each tensor: size, data
        for tensor_bytes in tensor_bytes_list:
            self.push_socket.send_multipart([
                len(tensor_bytes).to_bytes(8, 'big'),  # tensor size
                tensor_bytes,  # tensor data
            ])
        
        logger.debug(
            f"Stage {self.stage_idx} sent tensor dict with {len(tensor_list)} tensors"
        )
    
    def recv_tensor_dict(
        self,
        src: int | None = None,
    ) -> dict[str, torch.Tensor | Any] | None:
        """Receive tensor dictionary from previous stage.
        
        Args:
            src: Source stage index (ignored, always receives from previous stage)
            
        Returns:
            Received tensor dictionary, or None on failure
        """
        if self.is_first:
            raise RuntimeError("First stage cannot receive tensor dict")
        
        if self.pull_socket is None:
            raise RuntimeError("Pull socket not initialized")
        
        try:
            # Receive metadata
            parts = self.pull_socket.recv_multipart()
            if len(parts) < 3:
                raise RuntimeError(
                    f"Invalid message format: expected at least 3 parts, got {len(parts)}"
                )
            
            metadata_size = int.from_bytes(parts[0], 'big')
            metadata_bytes = parts[1]
            num_tensors = int.from_bytes(parts[2], 'big')
            
            if len(metadata_bytes) != metadata_size:
                raise RuntimeError(
                    f"Metadata size mismatch: expected {metadata_size}, got {len(metadata_bytes)}"
                )
            
            # Deserialize metadata
            metadata_list = pickle.loads(metadata_bytes)
            
            # Receive all tensors
            tensor_list = []
            for _ in range(num_tensors):
                tensor_parts = self.pull_socket.recv_multipart()
                if len(tensor_parts) < 2:
                    raise RuntimeError("Invalid tensor message format")
                
                tensor_size = int.from_bytes(tensor_parts[0], 'big')
                tensor_bytes = tensor_parts[1]
                
                if len(tensor_bytes) != tensor_size:
                    raise RuntimeError(
                        f"Tensor size mismatch: expected {tensor_size}, got {len(tensor_bytes)}"
                    )
                
                # Deserialize tensor
                buffer = io.BytesIO(tensor_bytes)
                tensor = torch.load(buffer, map_location="cpu", weights_only=True)
                tensor_list.append(tensor)
            
            # Reconstruct tensor dictionary
            tensor_dict: dict[str, Any] = {}
            tensor_idx = 0
            
            for key, value in metadata_list:
                if isinstance(value, TensorMetadata):
                    # Get tensor from tensor list
                    if tensor_idx >= len(tensor_list):
                        raise RuntimeError(
                            f"Not enough tensors: expected at least {tensor_idx + 1}"
                        )
                    
                    tensor = tensor_list[tensor_idx]
                    
                    # Restore dtype from metadata (torch.load may change dtype)
                    if tensor.dtype != value.dtype:
                        logger.debug(
                            f"Stage {self.stage_idx}: Tensor {key} dtype mismatch: "
                            f"loaded {tensor.dtype}, expected {value.dtype}. Converting..."
                        )
                        tensor = tensor.to(dtype=value.dtype)
                    
                    # Move tensor to target device
                    if self.device.startswith("cuda"):
                        tensor = tensor.to(dtype=value.dtype, device=self.device)
                    elif self.device == "cpu":
                        tensor = tensor.to(dtype=value.dtype, device="cpu")
                    
                    # Final dtype verification
                    if tensor.dtype != value.dtype:
                        logger.error(
                            f"Stage {self.stage_idx}: Failed to restore dtype for {key}: "
                            f"got {tensor.dtype}, expected {value.dtype}"
                        )
                    
                    tensor_dict[key] = tensor
                    tensor_idx += 1
                else:
                    # Non-tensor value, add directly
                    tensor_dict[key] = value
            
            logger.debug(
                f"Stage {self.stage_idx} received tensor dict with {len(tensor_list)} tensors"
            )
            
            return tensor_dict
            
        except zmq.Again:
            logger.error(f"Stage {self.stage_idx} receive timeout")
            return None
        except Exception as e:
            logger.error(
                f"Stage {self.stage_idx} receive error: {e}", exc_info=True
            )
            return None
    
    def close(self) -> None:
        """Close sockets and context."""
        if self.push_socket:
            self.push_socket.close()
            self.push_socket = None
        if self.pull_socket:
            self.pull_socket.close()
            self.pull_socket = None
        if self.context:
            self.context.term()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

