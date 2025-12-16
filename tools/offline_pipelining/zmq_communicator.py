#!/usr/bin/env python3
"""
ZeroMQ通信接口，用于pipeline stages之间的tensor传递

实现类似vLLM的send_tensor_dict/recv_tensor_dict接口，但使用ZeroMQ替代PyTorch distributed
"""

import io
import pickle
from typing import Any, Dict, Optional

import torch
import zmq

from vllm.logger import init_logger

logger = init_logger(__name__)


class TensorMetadata:
    """Tensor元数据，用于描述tensor的形状、类型和设备信息"""

    def __init__(self, device: str, dtype: torch.dtype, size: torch.Size):
        self.device = device  # 设备类型，如 "cuda", "cpu"
        self.dtype = dtype
        self.size = size

    def __repr__(self):
        return f"TensorMetadata(device={self.device}, dtype={self.dtype}, size={self.size})"


def _split_tensor_dict(
    tensor_dict: Dict[str, torch.Tensor | Any],
) -> tuple[list[tuple[str, Any]], list[torch.Tensor]]:
    """将tensor字典拆分为元数据列表和tensor列表
    
    Args:
        tensor_dict: 包含tensor和其他对象的字典
        
    Returns:
        metadata_list: (key, value) 列表，tensor被替换为TensorMetadata
        tensor_list: tensor对象列表
    """
    metadata_list: list[tuple[str, Any]] = []
    tensor_list: list[torch.Tensor] = []
    
    for key, value in tensor_dict.items():
        if isinstance(value, torch.Tensor):
            # 只保存设备类型，不保存设备索引（接收端会设置）
            device = value.device.type
            metadata_list.append(
                (key, TensorMetadata(device, value.dtype, value.size()))
            )
            tensor_list.append(value)
        else:
            metadata_list.append((key, value))
    
    return metadata_list, tensor_list


class ZeroMQCommunicator:
    """ZeroMQ通信器，用于pipeline stages之间的tensor传递"""

    def __init__(
        self,
        stage_idx: int,
        num_stages: int,
        zmq_ports: list[int],
        device: str = "cpu",
        timeout_ms: int = -1,  # -1 means no timeout, keep long connection alive
    ):
        """初始化ZeroMQ通信器
        
        Args:
            stage_idx: 当前stage的索引
            num_stages: 总stage数量
            zmq_ports: ZeroMQ端口列表，长度为num_stages-1
            device: 目标设备
            timeout_ms: Socket超时时间（毫秒）
        """
        self.stage_idx = stage_idx
        self.num_stages = num_stages
        self.zmq_ports = zmq_ports
        self.device = device
        self.timeout_ms = timeout_ms
        
        self.is_first = stage_idx == 0
        self.is_last = stage_idx == num_stages - 1
        
        # 创建ZeroMQ上下文
        self.context = zmq.Context()
        
        # 设置sockets
        self.push_socket: Optional[zmq.Socket] = None
        self.pull_socket: Optional[zmq.Socket] = None
        # 反向通信sockets（用于自回归生成）
        self.backward_push_socket: Optional[zmq.Socket] = None
        self.backward_pull_socket: Optional[zmq.Socket] = None
        
        self._setup_sockets()
    
    def _setup_sockets(self):
        """设置ZeroMQ sockets"""
        if not self.is_last:
            # 非最后stage需要发送数据到下一个stage
            self.push_socket = self.context.socket(zmq.PUSH)
            # 连接到下一个stage的PULL socket
            # 注意：下一个stage会bind，当前stage需要connect
            # 但为了简化，我们让发送方bind，接收方connect
            # 实际上应该是：stage i bind到port i，stage i+1 connect到port i
            if self.stage_idx < len(self.zmq_ports):
                port = self.zmq_ports[self.stage_idx]
                self.push_socket.bind(f"tcp://*:{port}")
                logger.info(f"Stage {self.stage_idx} bound PUSH socket to port {port}")
        
        if not self.is_first:
            # 非第一个stage需要接收数据
            self.pull_socket = self.context.socket(zmq.PULL)
            # 连接到上一个stage的PUSH socket
            prev_port = self.zmq_ports[self.stage_idx - 1]
            self.pull_socket.connect(f"tcp://localhost:{prev_port}")
            # 设置接收超时
            # No timeout: keep long connection alive (RCVTIMEO not set, will block until data arrives)
            # self.pull_socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            logger.info(
                f"Stage {self.stage_idx} connected PULL socket to port {prev_port}"
            )
        
        # 设置反向通信sockets（用于自回归生成：最后stage -> stage 0）
        # 使用额外的端口范围：base_port + 10000
        backward_port = self.zmq_ports[0] + 10000 if self.zmq_ports else 15555
        if self.is_last:
            # 最后stage需要发送token ID回stage 0
            self.backward_push_socket = self.context.socket(zmq.PUSH)
            self.backward_push_socket.bind(f"tcp://*:{backward_port}")
            logger.info(f"Stage {self.stage_idx} bound backward PUSH socket to port {backward_port}")
        
        if self.is_first:
            # 第一个stage需要接收token ID
            self.backward_pull_socket = self.context.socket(zmq.PULL)
            self.backward_pull_socket.connect(f"tcp://localhost:{backward_port}")
            # No timeout: keep long connection alive (RCVTIMEO not set, will block until data arrives)
            # self.backward_pull_socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            logger.info(f"Stage {self.stage_idx} connected backward PULL socket to port {backward_port}")
    
    def send_tensor_dict(
        self,
        tensor_dict: Dict[str, torch.Tensor | Any],
        dst: Optional[int] = None,
    ) -> None:
        """发送tensor字典到下一个stage
        
        Args:
            tensor_dict: 要发送的tensor字典
            dst: 目标stage索引（对于pipeline，通常是下一个stage，可忽略）
        """
        if self.is_last:
            raise RuntimeError("Last stage cannot send tensor dict")
        
        if self.push_socket is None:
            raise RuntimeError("Push socket not initialized")
        
        # 拆分tensor字典为元数据和tensor列表
        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        
        # 序列化元数据
        metadata_bytes = pickle.dumps(metadata_list)
        
        # 序列化所有tensors
        tensor_bytes_list = []
        for tensor in tensor_list:
            # 将tensor移到CPU进行序列化
            buffer = io.BytesIO()
            torch.save(tensor.cpu(), buffer)
            tensor_bytes_list.append(buffer.getvalue())
        
        # 发送：先发送元数据大小，然后元数据，然后每个tensor的大小和内容
        # 使用多部分消息：metadata_size, metadata, tensor1_size, tensor1, ...
        self.push_socket.send_multipart([
            len(metadata_bytes).to_bytes(8, 'big'),  # metadata size (8 bytes)
            metadata_bytes,  # metadata
            len(tensor_bytes_list).to_bytes(8, 'big'),  # number of tensors
        ])
        
        # 发送每个tensor
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
        src: Optional[int] = None,
    ) -> Optional[Dict[str, torch.Tensor | Any]]:
        """从上一个stage接收tensor字典
        
        Args:
            src: 源stage索引（对于pipeline，通常是上一个stage，可忽略）
            
        Returns:
            接收到的tensor字典，如果失败返回None
        """
        if self.is_first:
            raise RuntimeError("First stage cannot receive tensor dict")
        
        if self.pull_socket is None:
            raise RuntimeError("Pull socket not initialized")
        
        try:
            # 接收元数据
            parts = self.pull_socket.recv_multipart()
            if len(parts) < 3:
                raise RuntimeError(f"Invalid message format: expected at least 3 parts, got {len(parts)}")
            
            metadata_size = int.from_bytes(parts[0], 'big')
            metadata_bytes = parts[1]
            num_tensors = int.from_bytes(parts[2], 'big')
            
            if len(metadata_bytes) != metadata_size:
                raise RuntimeError(
                    f"Metadata size mismatch: expected {metadata_size}, got {len(metadata_bytes)}"
                )
            
            # 反序列化元数据
            metadata_list = pickle.loads(metadata_bytes)
            
            # 接收所有tensors
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
                
                # 反序列化tensor
                buffer = io.BytesIO(tensor_bytes)
                tensor = torch.load(buffer, map_location="cpu", weights_only=True)
                tensor_list.append(tensor)
            
            # 重建tensor字典
            tensor_dict: Dict[str, Any] = {}
            tensor_idx = 0
            
            for key, value in metadata_list:
                if isinstance(value, TensorMetadata):
                    # 从tensor列表获取tensor
                    if tensor_idx >= len(tensor_list):
                        raise RuntimeError(f"Not enough tensors: expected at least {tensor_idx + 1}")
                    
                    tensor = tensor_list[tensor_idx]
                    
                    # CRITICAL: Ensure dtype matches metadata
                    # torch.save/load should preserve dtype, but when loading to CPU,
                    # some dtypes (like float16) might be converted to float32.
                    # We must restore the original dtype from metadata.
                    if tensor.dtype != value.dtype:
                        logger.debug(
                            f"Stage {self.stage_idx}: Tensor {key} dtype mismatch: "
                            f"loaded {tensor.dtype}, expected {value.dtype}. Converting..."
                        )
                        tensor = tensor.to(dtype=value.dtype)
                    
                    # 确保tensor在正确的设备上（先转换dtype，再移动设备）
                    # 注意：对于 float16/bfloat16，CPU 可能不支持，但 CUDA 支持
                    if self.device.startswith("cuda"):
                        tensor = tensor.to(dtype=value.dtype, device=self.device)
                    elif self.device == "cpu":
                        # CPU 上，float16 可能需要转换为 float32，但我们应该保持原始 dtype
                        # 如果原始 dtype 是 float16/bfloat16，保持它（即使 CPU 可能不支持某些操作）
                        tensor = tensor.to(dtype=value.dtype, device="cpu")
                    
                    # 最终验证 dtype
                    if tensor.dtype != value.dtype:
                        logger.error(
                            f"Stage {self.stage_idx}: Failed to restore dtype for {key}: "
                            f"got {tensor.dtype}, expected {value.dtype}"
                        )
                    
                    tensor_dict[key] = tensor
                    tensor_idx += 1
                else:
                    # 非tensor值直接添加
                    tensor_dict[key] = value
            
            logger.debug(
                f"Stage {self.stage_idx} received tensor dict with {len(tensor_list)} tensors"
            )
            
            return tensor_dict
            
        except zmq.Again:
            logger.error(f"Stage {self.stage_idx} receive timeout")
            return None
        except Exception as e:
            logger.error(f"Stage {self.stage_idx} receive error: {e}", exc_info=True)
            return None
    
    def send_token_id(self, token_id: int) -> None:
        """发送token ID（用于自回归生成的反向通信）
        
        Args:
            token_id: 要发送的token ID
        """
        if not self.is_last:
            raise RuntimeError("Only last stage can send token ID")
        
        if self.backward_push_socket is None:
            raise RuntimeError("Backward push socket not initialized")
        
        # 发送token ID
        self.backward_push_socket.send(token_id.to_bytes(4, 'big', signed=True))
        logger.debug(f"Stage {self.stage_idx} sent token ID {token_id}")
    
    def recv_token_id(self) -> Optional[int]:
        """接收token ID（用于自回归生成的反向通信）
        
        Returns:
            接收到的token ID，如果失败返回None
        """
        if not self.is_first:
            raise RuntimeError("Only first stage can receive token ID")
        
        if self.backward_pull_socket is None:
            raise RuntimeError("Backward pull socket not initialized")
        
        try:
            data = self.backward_pull_socket.recv()
            token_id = int.from_bytes(data, 'big', signed=True)
            logger.debug(f"Stage {self.stage_idx} received token ID {token_id}")
            return token_id
        except zmq.Again:
            logger.error(f"Stage {self.stage_idx} receive token ID timeout")
            return None
        except Exception as e:
            logger.error(f"Stage {self.stage_idx} receive token ID error: {e}", exc_info=True)
            return None
    
    def close(self):
        """关闭sockets和上下文"""
        if self.push_socket:
            self.push_socket.close()
            self.push_socket = None
        if self.pull_socket:
            self.pull_socket.close()
            self.pull_socket = None
        if self.backward_push_socket:
            self.backward_push_socket.close()
            self.backward_push_socket = None
        if self.backward_pull_socket:
            self.backward_pull_socket.close()
            self.backward_pull_socket = None
        if self.context:
            self.context.term()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

