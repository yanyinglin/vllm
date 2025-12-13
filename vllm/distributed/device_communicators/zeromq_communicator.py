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
import time
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
        next_stage_addr: str | None = None,  # "ip:port" format for PULL to connect (bind mode)
        prev_stage_addr: str | None = None,  # "ip:port" format for return PULL to connect (bind mode)
        local_bind_port: int | None = None,  # Port for PUSH socket to bind (bind mode)
        device: str = "cuda",
        timeout_ms: int = 300000,  # Default 5 minutes for external PP (stages may wait for requests)
    ):
        """Initialize ZeroMQ communicator.
        
        Args:
            stage_idx: Current stage index (0-based)
            num_stages: Total number of stages
            local_listen_port: Local port for PULL socket (legacy) or connect address (bind mode)
            next_stage_addr: Address of next stage Service in "ip:port" format for PULL socket to connect (bind mode)
            prev_stage_addr: Address of previous stage Service in "ip:port" format for return PULL socket to connect (bind mode, optional)
            local_bind_port: Local port to bind for PUSH socket (bind mode), allowing multiple receivers via Service
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

        # Sockets for return communication (last stage -> stage 0)
        self.return_push_socket: zmq.Socket | None = None
        self.return_pull_socket: zmq.Socket | None = None

        self._setup_sockets(local_listen_port, next_stage_addr, prev_stage_addr, local_bind_port)
    
    def _setup_sockets(
        self,
        local_listen_port: int | None,
        next_stage_addr: str | None,
        prev_stage_addr: str | None,
        local_bind_port: int | None,
    ) -> None:
        """Setup ZeroMQ sockets for communication in bind mode.
        
        Bind mode: PUSH sockets bind to ports, PULL sockets connect to Service addresses.
        This allows Kubernetes Service load balancing - multiple PULL instances can connect
        to the same PUSH Service endpoint.
        """
        # Forward direction: receive from previous stage (stage_{k-1} -> stage_k)
        # In bind mode: PULL socket connects to previous stage's PUSH Service
        if not self.is_first:
            if next_stage_addr is None:
                raise ValueError(
                    f"Stage {self.stage_idx}: next_stage_addr must be provided for non-first stages "
                    "(this is the address of previous stage's PUSH Service to connect to)"
                )
            
            self.pull_socket = self.context.socket(zmq.PULL)
            # Set LINGER to ensure messages are not lost on close
            self.pull_socket.setsockopt(zmq.LINGER, 1000)
            # Connect to previous stage's PUSH Service (bind mode)
            self.pull_socket.connect(f"tcp://{next_stage_addr}")
            self.pull_socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            logger.info(
                f"Stage {self.stage_idx} connecting PULL socket to previous stage Service {next_stage_addr}"
            )
            # Wait for connection to establish
            max_wait_time = 10.0
            start_time = time.time()
            connection_ready = False
            
            poller = zmq.Poller()
            poller.register(self.pull_socket, zmq.POLLIN)
            
            while time.time() - start_time < max_wait_time:
                events = dict(poller.poll(100))
                if self.pull_socket in events:
                    connection_ready = True
                    logger.info(
                        f"Stage {self.stage_idx} PULL socket connection established to {next_stage_addr}"
                    )
                    break
                time.sleep(0.2)
            
            if not connection_ready:
                logger.warning(
                    f"Stage {self.stage_idx} PULL socket connection to {next_stage_addr} "
                    f"may not be ready after {max_wait_time}s. "
                    "This is OK if the previous stage hasn't started yet - ZeroMQ will queue messages."
                )

        # Forward direction: send to next stage (stage_k -> stage_{k+1})
        # In bind mode: PUSH socket binds to local port, allowing multiple receivers to connect
        if not self.is_last:
            if local_bind_port is None:
                raise ValueError(
                    f"Stage {self.stage_idx}: local_bind_port must be provided for non-last stages "
                    "(this is the port for PUSH socket to bind, allowing next stage receivers to connect)"
                )
            
            self.push_socket = self.context.socket(zmq.PUSH)
            # Set LINGER to ensure messages are not lost on close
            self.push_socket.setsockopt(zmq.LINGER, 1000)
            # Bind to local port (bind mode) - allows multiple PULL sockets to connect via Service
            self.push_socket.bind(f"tcp://*:{local_bind_port}")
            logger.info(
                f"Stage {self.stage_idx} bound PUSH socket to port {local_bind_port} "
                "(multiple receivers can connect via Service)"
            )
            # Small delay to ensure socket is ready for incoming connections
            time.sleep(0.2)

        # Return path: last stage -> stage 0
        #
        # In bind mode:
        # - Last stage (sender) binds a PUSH socket on `local_bind_port` (return port)
        #   for the return path, allowing stage 0's PULL socket to connect.
        # - Stage 0 (receiver) connects a PULL socket to `prev_stage_addr` (last stage's return Service).
        if self.is_last and prev_stage_addr is not None:
            # Last stage binds PUSH socket for return path (bind mode)
            # Note: prev_stage_addr is used to determine the bind port in bind mode
            # Actually, we need a separate return_bind_port parameter
            # For now, use local_bind_port if available, otherwise derive from prev_stage_addr
            if local_bind_port is None:
                # Try to extract port from prev_stage_addr or use a default offset
                # For simplicity, require local_bind_port to be set for return path
                raise ValueError(
                    f"Stage {self.stage_idx}: local_bind_port must be provided for last stage "
                    "return path (bind mode) - this is the port for return PUSH socket to bind"
                )
            return_bind_port = local_bind_port
            self.return_push_socket = self.context.socket(zmq.PUSH)
            # Set LINGER to ensure messages are not lost on close
            self.return_push_socket.setsockopt(zmq.LINGER, 1000)
            self.return_push_socket.bind(f"tcp://*:{return_bind_port}")
            logger.info(
                f"Stage {self.stage_idx} bound RETURN PUSH socket to port {return_bind_port} "
                "for final results to stage 0 (bind mode)"
            )
            # Small delay to ensure socket is ready for incoming connections
            time.sleep(0.2)

        if self.is_first and not self.is_last and prev_stage_addr is not None:
            # Stage 0 connects PULL socket to last stage's return Service (bind mode)
            self.return_pull_socket = self.context.socket(zmq.PULL)
            # Set LINGER to ensure messages are not lost on close
            self.return_pull_socket.setsockopt(zmq.LINGER, 1000)
            self.return_pull_socket.connect(f"tcp://{prev_stage_addr}")
            self.return_pull_socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            logger.info(
                f"Stage {self.stage_idx} connecting RETURN PULL socket to {prev_stage_addr} "
                f"to receive final results from last stage (bind mode)"
            )
            # Wait for connection to establish
            max_wait_time = 10.0
            start_time = time.time()
            connection_ready = False
            
            poller = zmq.Poller()
            poller.register(self.return_pull_socket, zmq.POLLIN)
            
            while time.time() - start_time < max_wait_time:
                events = dict(poller.poll(100))
                if self.return_pull_socket in events:
                    connection_ready = True
                    logger.info(
                        f"Stage {self.stage_idx} RETURN PULL socket connection established to {prev_stage_addr}"
                    )
                    break
                time.sleep(0.2)
            
            if not connection_ready:
                logger.warning(
                    f"Stage {self.stage_idx} RETURN PULL socket connection to {prev_stage_addr} "
                    f"may not be ready after {max_wait_time}s. "
                    "This is OK if the last stage hasn't started yet - ZeroMQ will queue messages."
                )
    
    def send_tensor_dict(
        self,
        tensor_dict: dict[str, torch.Tensor | Any],
        dst: int | None = None,
    ) -> None:
        """Send tensor dictionary.

        In external PP mode this is used for:
        - Forward path: non-last stages send to the next stage.
        - Return path: the last stage sends final results back to stage 0.

        Args:
            tensor_dict: Dictionary containing tensors and other objects
            dst: Destination stage index (ignored in external PP mode)
        """
        # Determine which socket to use based on stage role.
        if self.is_last:
            # Last stage sends results back to stage 0 over the return path.
            if self.return_push_socket is None:
                raise RuntimeError(
                    f"Stage {self.stage_idx}: return PUSH socket not initialized"
                )
            socket = self.return_push_socket
        else:
            # Non-last stages send forward to the next stage.
            if self.push_socket is None:
                raise RuntimeError(
                    f"Stage {self.stage_idx}: forward PUSH socket not initialized"
                )
            socket = self.push_socket

        # Split tensor dict into metadata and tensors
        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        
        # Debug: Log what we're sending
        tensor_keys = [k for k, v in tensor_dict.items() if isinstance(v, torch.Tensor)]
        non_tensor_keys = [k for k, v in tensor_dict.items() if not isinstance(v, torch.Tensor)]
        logger.info(
            f"Stage {self.stage_idx}: Sending tensor dict - "
            f"tensor keys: {tensor_keys}, non-tensor keys: {non_tensor_keys}, "
            f"total tensors: {len(tensor_list)}"
        )
        # CRITICAL: Verify hidden_states is a tensor before sending
        if "hidden_states" in tensor_dict:
            hidden_states = tensor_dict["hidden_states"]
            if not isinstance(hidden_states, torch.Tensor):
                raise RuntimeError(
                    f"Stage {self.stage_idx}: CRITICAL BUG - 'hidden_states' is not a tensor before sending! "
                    f"Got type {type(hidden_states)}, value: {str(hidden_states)[:200]}. "
                    f"tensor_dict keys: {list(tensor_dict.keys())}"
                )
            logger.info(
                f"Stage {self.stage_idx}: Verified hidden_states is tensor: shape={hidden_states.shape}, "
                f"dtype={hidden_states.dtype}, device={hidden_states.device}"
            )

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
        socket.send_multipart(
            [
            len(metadata_bytes).to_bytes(8, 'big'),  # metadata size (8 bytes)
            metadata_bytes,  # metadata
            len(tensor_bytes_list).to_bytes(8, 'big'),  # number of tensors
            ]
        )

        # Send each tensor: size, data
        for tensor_bytes in tensor_bytes_list:
            socket.send_multipart(
                [
                    len(tensor_bytes).to_bytes(8, 'big'),  # tensor size
                    tensor_bytes,  # tensor data
                ]
            )

        logger.debug(
            f"Stage {self.stage_idx} sent tensor dict with {len(tensor_list)} tensors"
        )
    
    def recv_tensor_dict(
        self,
        src: int | None = None,
    ) -> dict[str, torch.Tensor | Any] | None:
        """Receive tensor dictionary.

        In external PP mode this is used for:
        - Forward path: non-first stages receive from the previous stage.
        - Return path: stage 0 receives final results from the last stage.

        Args:
            src: Source stage index (ignored in external PP mode)

        Returns:
            Received tensor dictionary, or None on failure
        """
        # Determine which socket to use based on stage role.
        if self.is_first:
            # Stage 0 receives from the last stage over the return path.
            # In bind mode: stage 0 connects PULL socket to last stage's return Service
            if self.return_pull_socket is None:
                raise RuntimeError(
                    f"Stage {self.stage_idx}: return PULL socket not initialized"
                )
            socket = self.return_pull_socket
        else:
            # Non-first stages receive forward from previous stage.
            if self.pull_socket is None:
                raise RuntimeError(
                    f"Stage {self.stage_idx}: forward PULL socket not initialized"
                )
            socket = self.pull_socket

        try:
            # Receive metadata
            parts = socket.recv_multipart()
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
            try:
                metadata_list = pickle.loads(metadata_bytes)
            except Exception as e:
                raise RuntimeError(
                    f"Stage {self.stage_idx}: Failed to deserialize metadata: {e}. "
                    f"Metadata bytes length: {len(metadata_bytes)}"
                )
            
            # Validate metadata structure
            if not isinstance(metadata_list, list):
                raise RuntimeError(
                    f"Stage {self.stage_idx}: Invalid metadata format - expected list, "
                    f"got {type(metadata_list)}"
                )
            
            # Validate each metadata entry is a tuple with 2 elements
            for i, entry in enumerate(metadata_list):
                if not isinstance(entry, tuple) or len(entry) != 2:
                    raise RuntimeError(
                        f"Stage {self.stage_idx}: Invalid metadata entry at index {i}: "
                        f"expected tuple of (key, value), got {type(entry)}: {entry}"
                    )
                key, value = entry
                if not isinstance(key, str):
                    raise RuntimeError(
                        f"Stage {self.stage_idx}: Invalid metadata key at index {i}: "
                        f"expected str, got {type(key)}: {key}"
                    )
                
                # CRITICAL: Check if hidden_states is incorrectly stored as a non-tensor in metadata
                if key == "hidden_states" and not isinstance(value, TensorMetadata):
                    raise RuntimeError(
                        f"Stage {self.stage_idx}: CRITICAL BUG - 'hidden_states' in metadata is not a TensorMetadata! "
                        f"Got type {type(value)}, value: {str(value)[:200]}. "
                        f"This indicates a serialization bug in the sender. "
                        f"All metadata entries: {[(k, type(v).__name__) for k, v in metadata_list]}"
                    )
            
            # Receive all tensors
            tensor_list = []
            for _ in range(num_tensors):
                tensor_parts = socket.recv_multipart()
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
                # Note: weights_only=True may be too restrictive for some tensor serialization
                # If it fails, try without weights_only (but this is less secure)
                try:
                    loaded_obj = torch.load(buffer, map_location="cpu", weights_only=True)
                except Exception as e:
                    # If weights_only fails, try without it (for debugging)
                    buffer.seek(0)
                    logger.warning(
                        f"Stage {self.stage_idx}: torch.load with weights_only=True failed: {e}. "
                        "Retrying without weights_only..."
                    )
                    loaded_obj = torch.load(buffer, map_location="cpu", weights_only=False)
                
                # CRITICAL: Validate that deserialized object is actually a tensor
                if not isinstance(loaded_obj, torch.Tensor):
                    raise RuntimeError(
                        f"Stage {self.stage_idx}: Deserialization failed - expected torch.Tensor, "
                        f"got {type(loaded_obj)}. This indicates corrupted tensor data or "
                        f"serialization mismatch. Object value: {str(loaded_obj)[:200]}"
                    )
                
                tensor_list.append(loaded_obj)
            
            # Reconstruct tensor dictionary
            tensor_dict: dict[str, Any] = {}
            tensor_idx = 0
            
            # Debug: Log metadata structure for troubleshooting
            logger.debug(
                f"Stage {self.stage_idx}: Reconstructing tensor dict from {len(metadata_list)} metadata entries, "
                f"{len(tensor_list)} tensors"
            )
            
            # Log metadata keys in order for debugging
            metadata_keys = [k for k, v in metadata_list]
            logger.info(
                f"Stage {self.stage_idx}: Metadata keys in order: {metadata_keys}"
            )
            
            # Log tensor metadata types for debugging
            tensor_metadata_keys = [k for k, v in metadata_list if isinstance(v, TensorMetadata)]
            logger.info(
                f"Stage {self.stage_idx}: Tensor metadata keys (should match tensor_list order): {tensor_metadata_keys}"
            )
            
            # CRITICAL: Check hidden_states in metadata before reconstruction
            for key, value in metadata_list:
                if key == "hidden_states":
                    if not isinstance(value, TensorMetadata):
                        raise RuntimeError(
                            f"Stage {self.stage_idx}: CRITICAL BUG - 'hidden_states' in metadata is not TensorMetadata! "
                            f"Got type {type(value)}, value: {str(value)[:200]}. "
                            f"This indicates a serialization bug in the sender. "
                            f"All metadata entries: {[(k, type(v).__name__) for k, v in metadata_list]}"
                        )
                    logger.info(
                        f"Stage {self.stage_idx}: Found hidden_states in metadata: TensorMetadata(shape={value.size}, "
                        f"dtype={value.dtype}, device={value.device})"
                    )
            
            for key, value in metadata_list:
                if isinstance(value, TensorMetadata):
                    # CRITICAL: Special check for hidden_states - it MUST be a tensor
                    if key == "hidden_states":
                        # Ensure we have enough tensors
                        if tensor_idx >= len(tensor_list):
                            raise RuntimeError(
                                f"Stage {self.stage_idx}: CRITICAL - Not enough tensors for 'hidden_states'! "
                                f"expected at least {tensor_idx + 1}, got {len(tensor_list)}. "
                                f"This indicates a serialization/deserialization mismatch. "
                                f"metadata_keys: {metadata_keys}, tensor_list length: {len(tensor_list)}"
                            )
                    
                    # Get tensor from tensor list
                    if tensor_idx >= len(tensor_list):
                        raise RuntimeError(
                            f"Stage {self.stage_idx}: Not enough tensors for key '{key}': "
                            f"expected at least {tensor_idx + 1}, got {len(tensor_list)}. "
                            f"This indicates a serialization/deserialization mismatch. "
                            f"metadata_keys: {metadata_keys}"
                        )
                    
                    tensor = tensor_list[tensor_idx]
                    logger.info(
                        f"Stage {self.stage_idx}: Reconstructing tensor '{key}' (idx={tensor_idx}): "
                        f"expected shape={value.size}, dtype={value.dtype}, device={value.device}, "
                        f"actual tensor type={type(tensor)}, shape={tensor.shape if isinstance(tensor, torch.Tensor) else 'N/A'}"
                    )
                    
                    # CRITICAL: Double-check that tensor is actually a tensor (defense in depth)
                    if not isinstance(tensor, torch.Tensor):
                        raise RuntimeError(
                            f"Stage {self.stage_idx}: Tensor reconstruction failed for key '{key}'. "
                            f"Expected torch.Tensor, got {type(tensor)}. "
                            f"This indicates a deserialization bug. tensor_idx={tensor_idx}, "
                            f"tensor_list length={len(tensor_list)}, "
                            f"tensor value: {str(tensor)[:200] if hasattr(tensor, '__str__') else repr(tensor)[:200]}"
                        )
                    
                    # CRITICAL: Extra validation for hidden_states
                    if key == "hidden_states":
                        # Verify shape matches expected
                        if tensor.shape != value.size:
                            raise RuntimeError(
                                f"Stage {self.stage_idx}: CRITICAL - 'hidden_states' shape mismatch! "
                                f"Expected {value.size}, got {tensor.shape}. "
                                f"This indicates tensor corruption or mismatch."
                            )
                    
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
                    logger.info(
                        f"Stage {self.stage_idx}: Assigned tensor to tensor_dict['{key}']: "
                        f"type={type(tensor_dict[key])}, shape={tensor_dict[key].shape if isinstance(tensor_dict[key], torch.Tensor) else 'N/A'}"
                    )
                    tensor_idx += 1
                else:
                    # Non-tensor value, add directly
                    # CRITICAL: Validate that we're not accidentally assigning a list to hidden_states
                    if key == "hidden_states":
                        raise RuntimeError(
                            f"Stage {self.stage_idx}: CRITICAL BUG - 'hidden_states' key found with non-tensor value! "
                            f"This should never happen - hidden_states must be a tensor. "
                            f"Got type {type(value)}, value: {str(value)[:200]}. "
                            f"This indicates a metadata corruption or key-value mismatch. "
                            f"metadata_keys: {metadata_keys}"
                        )
                    logger.info(
                        f"Stage {self.stage_idx}: Adding non-tensor value for key '{key}': "
                        f"type={type(value)}, value={str(value)[:100] if not isinstance(value, (list, dict)) else f'{type(value).__name__} with {len(value)} items'}"
                    )
                    tensor_dict[key] = value
                    logger.info(
                        f"Stage {self.stage_idx}: Assigned non-tensor to tensor_dict['{key}']: "
                        f"type={type(tensor_dict[key])}"
                    )
            
            # Final validation: ensure critical keys have correct types
            logger.info(
                f"Stage {self.stage_idx}: Final tensor_dict keys: {list(tensor_dict.keys())}, "
                f"hidden_states type: {type(tensor_dict.get('hidden_states', 'NOT_FOUND'))}"
            )
            if "hidden_states" in tensor_dict:
                if not isinstance(tensor_dict["hidden_states"], torch.Tensor):
                    # Additional debugging: check if request_ids has the same value
                    request_ids_value = tensor_dict.get("request_ids")
                    if isinstance(request_ids_value, list) and len(request_ids_value) == len(tensor_dict["hidden_states"]) if isinstance(tensor_dict["hidden_states"], list) else False:
                        if request_ids_value == tensor_dict["hidden_states"]:
                            raise RuntimeError(
                                f"Stage {self.stage_idx}: CRITICAL BUG - 'hidden_states' has the same value as 'request_ids'! "
                                f"This indicates a key-value mismatch in deserialization. "
                                f"hidden_states value: {tensor_dict['hidden_states']}, "
                                f"request_ids value: {request_ids_value}, "
                                f"metadata_keys: {metadata_keys}, "
                                f"tensor_dict keys: {list(tensor_dict.keys())}"
                            )
                    raise RuntimeError(
                        f"Stage {self.stage_idx}: CRITICAL - 'hidden_states' is not a tensor after reconstruction! "
                        f"Got type {type(tensor_dict['hidden_states'])}, value: {str(tensor_dict['hidden_states'])[:200]}. "
                        f"This indicates a serious deserialization bug. "
                        f"tensor_dict keys: {list(tensor_dict.keys())}, "
                        f"tensor_list length: {len(tensor_list)}, "
                        f"metadata_keys: {metadata_keys}"
                    )
            
            logger.info(
                f"Stage {self.stage_idx} received tensor dict with {len(tensor_list)} tensors, "
                f"keys: {list(tensor_dict.keys())}"
            )
            
            # Final check before returning
            if "hidden_states" in tensor_dict:
                hs = tensor_dict["hidden_states"]
                logger.info(
                    f"Stage {self.stage_idx}: Final check - hidden_states type before return: {type(hs)}, "
                    f"is_tensor={isinstance(hs, torch.Tensor)}, "
                    f"shape={hs.shape if isinstance(hs, torch.Tensor) else 'N/A'}"
                )
                if not isinstance(hs, torch.Tensor):
                    raise RuntimeError(
                        f"Stage {self.stage_idx}: CRITICAL - hidden_states is not a tensor before return! "
                        f"Type: {type(hs)}, value: {str(hs)[:200]}"
                    )
            
            return tensor_dict
            
        except zmq.Again:
            logger.error(
                f"Stage {self.stage_idx} receive timeout after {self.timeout_ms}ms. "
                "This may indicate a stage failure or network issue."
            )
            return None
        except Exception as e:
            logger.error(
                f"Stage {self.stage_idx} receive error: {e}", exc_info=True
            )
            # Re-raise as RuntimeError so callers can handle it properly
            raise RuntimeError(
                f"Stage {self.stage_idx} failed to receive tensor dict: {e}"
            ) from e
    
    def close(self) -> None:
        """Close sockets and context."""
        if self.push_socket:
            self.push_socket.close()
            self.push_socket = None
        if self.pull_socket:
            self.pull_socket.close()
            self.pull_socket = None
        if self.return_push_socket:
            self.return_push_socket.close()
            self.return_push_socket = None
        if self.return_pull_socket:
            self.return_pull_socket.close()
            self.return_pull_socket = None
        if self.context:
            self.context.term()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

