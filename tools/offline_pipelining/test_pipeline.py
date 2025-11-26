#!/usr/bin/env python3
"""
vLLM离线流水线测试工具

加载导出的pipeline阶段并使用ZeroMQ进行分布式推理测试。
"""

import argparse
import io
import json
import multiprocessing
import os
import pickle
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import zmq
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from vllm.logger import init_logger

# 导入zmq_communicator（使用相对导入）
try:
    from .zmq_communicator import ZeroMQCommunicator
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    from zmq_communicator import ZeroMQCommunicator

logger = init_logger(__name__)


class PipelineStage:
    """表示一个pipeline stage"""

    def __init__(
        self,
        stage_dir: Path,
        stage_idx: int,
        device: str = "cpu",
        num_stages: int = 1,
    ):
        self.stage_dir = stage_dir
        self.stage_idx = stage_idx
        self.device = device
        self.num_stages = num_stages
        self.model = None
        self.config = None
        self.tokenizer = None

    def load(self):
        """加载stage模型（使用HuggingFace标准加载方式）
        
        直接使用 AutoModelForCausalLM.from_pretrained() 加载，因为模型已保存为HuggingFace格式。
        这样可以：
        1. 自动加载 config.json
        2. 自动处理权重文件（model.safetensors 或 pytorch_model.bin）
        3. 自动处理键名匹配（HuggingFace内部已处理）
        4. 利用操作系统的page cache（文件内容会被缓存到内存）
        
        注意：Linux操作系统的page cache会自动缓存文件内容：
        - 第一次加载：从磁盘读取，文件被缓存到page cache（内存）
        - 后续加载：从page cache读取（内存速度），无需访问磁盘
        - 多进程共享：所有进程共享同一份page cache，文件只从磁盘读取一次
        """
        logger.info(f"Loading stage {self.stage_idx} from {self.stage_dir} (using HuggingFace from_pretrained)")
        
        # 检查是否已经加载过（避免重复加载）
        if self.model is not None:
            logger.info(f"Stage {self.stage_idx} already loaded, skipping")
            return

        # 加载配置（用于后续使用）
        config_path = self.stage_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # 直接使用 HuggingFace 的 from_pretrained 加载
        # 这会自动处理 config.json 和权重文件的加载
        from transformers import AutoModelForCausalLM
        
        # 从config获取dtype信息（优先使用dtype，向后兼容torch_dtype）
        if "dtype" in self.config:
            dtype_str = self.config["dtype"]
        elif "torch_dtype" in self.config:
            dtype_str = self.config["torch_dtype"]
        else:
            dtype_str = "float16"
        
        if dtype_str == "float16" or dtype_str == "torch.float16":
            torch_dtype = torch.float16
        elif dtype_str == "bfloat16" or dtype_str == "torch.bfloat16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16
        
        # 使用 HuggingFace 方式加载：from_config + 手动加载权重
        # 注意：不能直接使用 from_pretrained，因为它会检查 state dict 的完整性
        # 而 pipeline 拆分后的模型是不完整的（只有部分层）
        from transformers import AutoConfig
        
        try:
            # 1. 加载 config 并创建模型结构
            config = AutoConfig.from_pretrained(
                str(self.stage_dir),
                trust_remote_code=True,
            )
            
            # 2. 从 config 创建模型（不加载权重）
            self.model = AutoModelForCausalLM.from_config(
                config,
                trust_remote_code=True,
                dtype=torch_dtype,  # 使用dtype代替已弃用的torch_dtype
            )
            
            # 3. 手动加载权重文件
            weights_file = None
            if (self.stage_dir / "model.safetensors").exists():
                weights_file = str(self.stage_dir / "model.safetensors")
            elif (self.stage_dir / "pytorch_model.bin").exists():
                weights_file = str(self.stage_dir / "pytorch_model.bin")
            
            if weights_file:
                logger.info(f"Loading weights from {weights_file}")
                # 利用操作系统的 page cache（文件内容会被缓存到内存）
                if weights_file.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    state_dict = load_file(weights_file, device="cpu")
                else:
                    state_dict = torch.load(weights_file, map_location="cpu", weights_only=True)
                
                # 4. 加载权重（使用 strict=False 允许部分权重不匹配）
                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    logger.debug(f"Stage {self.stage_idx} missing keys (first 10): {missing_keys[:10]}")
                if unexpected_keys:
                    logger.debug(f"Stage {self.stage_idx} unexpected keys (first 10): {unexpected_keys[:10]}")
                logger.info(f"Stage {self.stage_idx} loaded {len(state_dict)} weights")
            else:
                logger.warning(f"Stage {self.stage_idx} no weights file found, using random initialization")
            
            # 移动到目标设备
            if self.device != "cpu":
                self.model = self.model.to(self.device)
            
            # 设置为评估模式
            self.model.eval()
            logger.info(f"Stage {self.stage_idx} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load stage {self.stage_idx}: {e}", exc_info=True)
            raise

        # 加载tokenizer（仅第一阶段）
        if self.stage_idx == 0:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(self.stage_dir), trust_remote_code=True
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}")

    def forward(self, inputs, position_ids=None, attention_mask=None, past_key_values=None, use_cache=True):
        """前向传播
        
        Args:
            inputs: 输入（可以是tensor或dict）
            position_ids: position IDs
            attention_mask: attention mask
            past_key_values: 之前的KV cache（用于自回归生成）
            use_cache: 是否使用和返回KV cache
        """
        if self.model is None:
            raise RuntimeError(f"Stage {self.stage_idx} not loaded")

        with torch.no_grad():
            if isinstance(inputs, torch.Tensor):
                # 输入是tensor（hidden states）- 中间stage或最后stage
                hidden_states = inputs
                
                # 创建position_ids和attention_mask
                batch_size, seq_length = hidden_states.shape[:2]
                if position_ids is None:
                    position_ids = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
                if attention_mask is None:
                    attention_mask = torch.ones(
                        (batch_size, seq_length),
                        dtype=torch.bool,
                        device=hidden_states.device
                    )
                
                # 使用model.model的forward方法，传入inputs_embeds
                # 这样可以正确处理RoPE等
                try:
                    outputs = self.model.model(
                        inputs_embeds=hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                    )
                    
                    # 提取hidden states和past_key_values
                    if hasattr(outputs, "last_hidden_state"):
                        hidden_states = outputs.last_hidden_state
                        new_past_key_values = getattr(outputs, "past_key_values", None)
                    elif isinstance(outputs, tuple):
                        # outputs可能是 (hidden_states, past_key_values) 或 (hidden_states,)
                        hidden_states = outputs[0]
                        new_past_key_values = outputs[1] if len(outputs) > 1 and use_cache else None
                    else:
                        hidden_states = outputs
                        new_past_key_values = None
                except Exception as e:
                    logger.error(f"Stage {self.stage_idx} model forward failed: {e}")
                    # 回退到手动调用layers（不支持KV cache）
                    if hasattr(self.model.model, "layers"):
                        layers = self.model.model.layers
                    elif hasattr(self.model.model, "h"):
                        layers = self.model.model.h
                    else:
                        raise RuntimeError(f"Stage {self.stage_idx}: No layers found in model")
                    
                    # 手动调用layers（需要正确处理RoPE）
                    for layer in layers:
                        # 需要从config获取rope信息
                        layer_output = layer(
                            hidden_states,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                        )
                        hidden_states = layer_output[0] if isinstance(layer_output, tuple) else layer_output
                    new_past_key_values = None
                
                # 如果是最后stage，还需要通过norm和lm_head
                if self.stage_idx == self.num_stages - 1:
                    # 通过norm
                    if hasattr(self.model.model, "norm"):
                        hidden_states = self.model.model.norm(hidden_states)
                    # 通过lm_head
                    if hasattr(self.model, "lm_head"):
                        logits = self.model.lm_head(hidden_states)
                        if use_cache and new_past_key_values is not None:
                            return logits, new_past_key_values
                        return logits
                
                # 返回hidden states和past_key_values（如果使用cache）
                if use_cache and new_past_key_values is not None:
                    return hidden_states, new_past_key_values
                return hidden_states
            else:
                # 输入是input_ids - 第一阶段
                # 使用模型的forward方法，但只返回hidden_states
                input_ids = inputs.get("input_ids")
                if input_ids is None:
                    raise ValueError("Stage 0 requires input_ids")
                
                # 准备forward参数
                forward_kwargs = {"input_ids": input_ids}
                if position_ids is not None:
                    forward_kwargs["position_ids"] = position_ids
                if attention_mask is not None:
                    forward_kwargs["attention_mask"] = attention_mask
                elif "attention_mask" in inputs:
                    forward_kwargs["attention_mask"] = inputs["attention_mask"]
                
                # 支持past_key_values（用于自回归生成）
                if past_key_values is not None:
                    forward_kwargs["past_key_values"] = past_key_values
                forward_kwargs["use_cache"] = use_cache
                
                # 调用模型forward，但只获取hidden_states
                # 使用output_hidden_states=True来获取中间hidden states
                forward_kwargs["output_hidden_states"] = True
                outputs = self.model(**forward_kwargs)
                
                # 获取最后一层的hidden states（在layers之后，norm之前）
                if hasattr(outputs, "hidden_states") and outputs.hidden_states:
                    # hidden_states是一个tuple，最后一个是在所有layers之后
                    hidden_states = outputs.hidden_states[-1]
                elif hasattr(outputs, "last_hidden_state"):
                    hidden_states = outputs.last_hidden_state
                else:
                    # 如果没有hidden_states，手动提取
                    # 通过embedding
                    if hasattr(self.model.model, "embed_tokens"):
                        hidden_states = self.model.model.embed_tokens(input_ids)
                    else:
                        raise RuntimeError("Stage 0: No embed_tokens found")
                    
                    # 通过layers（不支持KV cache的回退路径）
                    layers = None
                    if hasattr(self.model.model, "layers"):
                        layers = self.model.model.layers
                    elif hasattr(self.model.model, "h"):
                        layers = self.model.model.h
                    
                    if layers is None:
                        raise RuntimeError("Stage 0: No layers found")
                    
                    # 创建position_ids
                    seq_length = input_ids.shape[1]
                    if position_ids is None:
                        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)
                    
                    # 创建attention_mask
                    if attention_mask is None:
                        attention_mask = torch.ones(
                            (input_ids.shape[0], seq_length),
                            dtype=torch.bool,
                            device=input_ids.device
                        )
                    
                    for layer in layers:
                        layer_output = layer(hidden_states, position_ids=position_ids, attention_mask=attention_mask)
                        hidden_states = layer_output[0] if isinstance(layer_output, tuple) else layer_output
                    # 手动提取路径不支持KV cache
                    new_past_key_values = None
                    # 返回hidden states（不使用cache）
                    return hidden_states
                
                # 提取past_key_values（如果使用cache）
                new_past_key_values = getattr(outputs, "past_key_values", None)
                
                # 返回hidden states和past_key_values（如果使用cache）
                if use_cache and new_past_key_values is not None:
                    return hidden_states, new_past_key_values
                return hidden_states


class ZeroMQPipeline:
    """使用ZeroMQ进行pipeline阶段间通信"""

    def __init__(
        self,
        pipeline_dir: Path,
        num_stages: int,
        zmq_ports: Optional[list[int]] = None,
        device: str = "cpu",
    ):
        self.pipeline_dir = Path(pipeline_dir)
        self.num_stages = num_stages
        self.device = device
        self.stages = []

        # 设置ZeroMQ端口
        if zmq_ports is None:
            base_port = 5555
            self.zmq_ports = [base_port + i for i in range(num_stages - 1)]
        else:
            if len(zmq_ports) != num_stages - 1:
                raise ValueError(
                    f"Expected {num_stages - 1} ports, got {len(zmq_ports)}"
                )
            self.zmq_ports = zmq_ports

        # 加载所有stages，分布到不同GPU
        for i in range(num_stages):
            stage_dir = self.pipeline_dir / f"stage_{i}"
            if not stage_dir.exists():
                raise FileNotFoundError(f"Stage directory not found: {stage_dir}")

            # 分配GPU：如果有多个GPU，每个stage使用不同的GPU
            if device.startswith("cuda") and torch.cuda.device_count() > 1:
                gpu_id = i % torch.cuda.device_count()
                stage_device = f"cuda:{gpu_id}"
                logger.info(f"Stage {i} will use {stage_device}")
            else:
                stage_device = device
            
            stage = PipelineStage(stage_dir, i, device=stage_device, num_stages=num_stages)
            stage.load()
            self.stages.append(stage)

        # ZeroMQ上下文
        self.context = zmq.Context()

    def serialize_tensor(self, tensor: torch.Tensor) -> bytes:
        """序列化tensor为bytes"""
        buffer = io.BytesIO()
        torch.save(tensor.cpu(), buffer)
        return buffer.getvalue()

    def deserialize_tensor(self, data: bytes) -> torch.Tensor:
        """从bytes反序列化tensor"""
        buffer = io.BytesIO(data)
        tensor = torch.load(buffer, map_location=self.device, weights_only=True)
        return tensor

    def run_stage(self, stage_idx: int):
        """运行单个stage（用于多线程）"""
        stage = self.stages[stage_idx]
        is_first = stage_idx == 0
        is_last = stage_idx == self.num_stages - 1

        if is_first:
            # 第一阶段：接收input_ids，发送hidden states
            socket = self.context.socket(zmq.PUSH)
            socket.bind(f"tcp://*:{self.zmq_ports[0]}")
            logger.info(f"Stage {stage_idx} bound to port {self.zmq_ports[0]}")
        elif is_last:
            # 最后阶段：接收hidden states，输出logits
            socket = self.context.socket(zmq.PULL)
            socket.connect(f"tcp://localhost:{self.zmq_ports[stage_idx - 1]}")
            logger.info(
                f"Stage {stage_idx} connected to port {self.zmq_ports[stage_idx - 1]}"
            )
        else:
            # 中间阶段：接收hidden states，发送hidden states
            pull_socket = self.context.socket(zmq.PULL)
            pull_socket.connect(f"tcp://localhost:{self.zmq_ports[stage_idx - 1]}")
            push_socket = self.context.socket(zmq.PUSH)
            push_socket.bind(f"tcp://*:{self.zmq_ports[stage_idx]}")
            logger.info(
                f"Stage {stage_idx} connected to port {self.zmq_ports[stage_idx - 1]} "
                f"and bound to port {self.zmq_ports[stage_idx]}"
            )

        return socket if is_first or is_last else (pull_socket, push_socket)

    def inference(self, input_text: str) -> str:
        """执行推理"""
        logger.info(f"Running inference on input: {input_text}")

        # 第一阶段：tokenization
        tokenizer = self.stages[0].tokenizer
        if tokenizer is None:
            raise RuntimeError("Tokenizer not available in first stage")

        inputs = tokenizer(input_text, return_tensors="pt", padding=True)
        # 将inputs移到stage 0的device
        stage0_device = self.stages[0].device
        inputs = {k: v.to(stage0_device) for k, v in inputs.items()}
        
        # 创建position_ids和attention_mask
        seq_length = inputs["input_ids"].shape[1]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=stage0_device).unsqueeze(0)
        attention_mask = inputs.get("attention_mask", torch.ones(
            (inputs["input_ids"].shape[0], seq_length),
            dtype=torch.bool,
            device=stage0_device
        ))

        # 如果是单stage，直接处理
        if self.num_stages == 1:
            logger.info("Running single stage inference...")
            outputs = self.stages[0].forward(inputs)
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs
        else:
            # 多stage pipeline推理
            # 使用ZeroMQ在stages之间传递数据
            # 对于测试，我们使用REQ/REP模式，顺序执行每个stage

            hidden_states = None

            # Stage 0: 处理input_ids
            logger.info("Running stage 0...")
            stage0_output = self.stages[0].forward(
                inputs, 
                position_ids=position_ids,
                attention_mask=attention_mask
            )

            # 提取hidden states
            if isinstance(stage0_output, torch.Tensor):
                hidden_states = stage0_output
            elif hasattr(stage0_output, "last_hidden_state"):
                hidden_states = stage0_output.last_hidden_state
            else:
                hidden_states = stage0_output[0] if isinstance(stage0_output, tuple) else stage0_output
            
            # 确保hidden_states在正确的device上（下一个stage的device）
            if self.num_stages > 1:
                next_stage_device = self.stages[1].device
                hidden_states = hidden_states.to(next_stage_device)

            # 中间和最后stages: 通过ZeroMQ传递数据（模拟分布式通信）
            # 使用序列化/反序列化来验证ZeroMQ通信机制
            for stage_idx in range(1, self.num_stages):
                logger.info(f"Processing stage {stage_idx}...")

                # 序列化hidden states（模拟通过ZeroMQ发送）
                data = self.serialize_tensor(hidden_states)
                logger.info(
                    f"Serialized hidden states for stage {stage_idx} "
                    f"(size: {len(data)} bytes)"
                )

                # 反序列化（模拟通过ZeroMQ接收）
                received_hidden_states = self.deserialize_tensor(data)
                logger.info(
                    f"Deserialized hidden states for stage {stage_idx} "
                    f"(shape: {received_hidden_states.shape})"
                )
                
                # 确保hidden_states在当前stage的device上
                current_stage_device = self.stages[stage_idx].device
                received_hidden_states = received_hidden_states.to(current_stage_device)

                # 运行当前stage（中间stage不需要position_ids，最后stage也不需要）
                stage_output = self.stages[stage_idx].forward(received_hidden_states)

                # 提取输出
                if hasattr(stage_output, "last_hidden_state"):
                    hidden_states = stage_output.last_hidden_state
                elif hasattr(stage_output, "logits"):
                    hidden_states = stage_output.logits
                elif isinstance(stage_output, torch.Tensor):
                    hidden_states = stage_output
                else:
                    hidden_states = (
                        stage_output[0]
                        if isinstance(stage_output, tuple)
                        else stage_output
                    )

            # 最后stage的输出就是logits
            logits = hidden_states

        # 解码输出
        if hasattr(logits, "shape") and len(logits.shape) == 3:
            # 取最后一个token的logits
            next_token_logits = logits[0, -1, :]
        elif hasattr(logits, "shape") and len(logits.shape) == 2:
            next_token_logits = logits[-1, :]
        else:
            next_token_logits = logits

        next_token_id = torch.argmax(next_token_logits, dim=-1).item()
        output_text = tokenizer.decode([next_token_id], skip_special_tokens=True)

        logger.info(f"Generated token ID: {next_token_id}, text: {output_text}")
        return output_text


def stage_worker_process(
    stage_idx: int,
    pipeline_dir: str,
    num_stages: int,
    zmq_ports: list[int],
    device: str,
    input_queue: Optional[multiprocessing.Queue],
    output_queue: Optional[multiprocessing.Queue],
    max_new_tokens: int = 256,
):
    """Stage worker进程：加载模型并处理请求"""
    import os
    import sys
    
    try:
        # 设置当前进程的GPU
        if device.startswith("cuda") and torch.cuda.device_count() > 1:
            gpu_id = stage_idx % torch.cuda.device_count()
            stage_device = f"cuda:{gpu_id}"
            # 在spawn模式下，每个进程需要重新初始化CUDA
            # 直接使用device字符串，模型加载时会自动使用
        else:
            stage_device = device
        
        print(f"[Stage {stage_idx}] Worker process started on {stage_device} (PID: {os.getpid()})", file=sys.stderr, flush=True)
        logger.info(f"Stage {stage_idx} worker process started on {stage_device} (PID: {os.getpid()})")
        
        # 加载stage
        stage_dir = Path(pipeline_dir) / f"stage_{stage_idx}"
        print(f"[Stage {stage_idx}] Loading from {stage_dir}", file=sys.stderr, flush=True)
        logger.info(f"Stage {stage_idx} loading from {stage_dir}")
        
        stage = PipelineStage(stage_dir, stage_idx, device=stage_device, num_stages=num_stages)
        stage.load()
        print(f"[Stage {stage_idx}] Loaded successfully on {stage_device}", file=sys.stderr, flush=True)
        logger.info(f"Stage {stage_idx} loaded successfully on {stage_device}")
        
        # 设置ZeroMQ通信器
        communicator = ZeroMQCommunicator(
            stage_idx=stage_idx,
            num_stages=num_stages,
            zmq_ports=zmq_ports,
            device=stage_device,
            timeout_ms=300000,  # 5分钟超时
        )
        
        is_first = stage_idx == 0
        is_last = stage_idx == num_stages - 1
        
        if is_first:
            # 第一阶段：从input_queue接收，发送到下一个stage
            print(f"[Stage {stage_idx}] ZeroMQ communicator initialized", file=sys.stderr, flush=True)
            logger.info(f"Stage {stage_idx} ZeroMQ communicator initialized")
            
            # 等待输入（无限等待，直到收到数据或None）
            print(f"[Stage {stage_idx}] Waiting for input from queue...", file=sys.stderr, flush=True)
            logger.info(f"Stage {stage_idx} waiting for input...")
            
            # 处理初始输入
            try:
                input_data = input_queue.get(timeout=300)  # 增加超时时间
                if input_data is None:  # 结束信号
                    logger.info(f"Stage {stage_idx} received exit signal")
                    communicator.close()
                    return
                
                input_text = input_data
                logger.info(f"Stage {stage_idx} received input: {input_text}")
                
                # Tokenization
                inputs = stage.tokenizer(input_text, return_tensors="pt", padding=True)
                inputs = {k: v.to(stage_device) for k, v in inputs.items()}
                
                # 保存初始的input_ids，用于后续的自回归生成
                initial_input_ids = inputs["input_ids"].clone()
                seq_length = inputs["input_ids"].shape[1]
                
                position_ids = torch.arange(seq_length, dtype=torch.long, device=stage_device).unsqueeze(0)
                attention_mask = inputs.get("attention_mask", torch.ones(
                    (inputs["input_ids"].shape[0], seq_length),
                    dtype=torch.bool,
                    device=stage_device
                ))
                
                # Forward（初始prompt，不使用KV cache）
                forward_result = stage.forward(inputs, position_ids=position_ids, attention_mask=attention_mask, use_cache=True)
                if isinstance(forward_result, tuple):
                    hidden_states, past_key_values = forward_result
                else:
                    hidden_states = forward_result
                    past_key_values = None
                
                # 使用ZeroMQ通信器发送tensor dict
                tensor_dict = {"hidden_states": hidden_states}
                communicator.send_tensor_dict(tensor_dict)
                logger.info(f"Stage {stage_idx} sent initial hidden states to next stage")
                
                # 处理自回归生成：使用KV cache避免重新计算所有tokens
                generation_step = 0
                while True:
                    token_id = communicator.recv_token_id()
                    if token_id is None:
                        # 超时或错误，退出
                        logger.info(f"Stage {stage_idx} no more tokens to process (generation complete)")
                        break
                    
                    generation_step += 1
                    logger.info(f"Stage {stage_idx} received token ID {token_id} for generation step {generation_step}")
                    print(f"[Stage {stage_idx}] Processing token {token_id} for generation step {generation_step} (position {seq_length})", file=sys.stderr, flush=True)
                    
                    # 使用KV cache：只处理新token，使用之前的past_key_values
                    new_token_tensor = torch.tensor([[token_id]], dtype=torch.long, device=stage_device)
                    new_position_ids = torch.tensor([[seq_length]], dtype=torch.long, device=stage_device)
                    new_attention_mask = torch.ones((1, 1), dtype=torch.bool, device=stage_device)
                    
                    # 通过stage 0的forward处理新token（使用KV cache）
                    new_inputs = {"input_ids": new_token_tensor}
                    forward_result = stage.forward(
                        new_inputs,
                        position_ids=new_position_ids,
                        attention_mask=new_attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    
                    if isinstance(forward_result, tuple):
                        new_hidden_states, past_key_values = forward_result
                    else:
                        new_hidden_states = forward_result
                        # 如果没有返回past_key_values，说明KV cache不可用，回退到重新计算所有tokens
                        logger.warning(f"Stage {stage_idx} KV cache not available, falling back to full recomputation")
                        updated_input_ids = torch.cat([initial_input_ids, new_token_tensor], dim=1)
                        seq_length = updated_input_ids.shape[1]
                        new_position_ids = torch.arange(seq_length, dtype=torch.long, device=stage_device).unsqueeze(0)
                        new_attention_mask = torch.ones((1, seq_length), dtype=torch.bool, device=stage_device)
                        new_inputs = {"input_ids": updated_input_ids}
                        new_hidden_states = stage.forward(
                            new_inputs,
                            position_ids=new_position_ids,
                            attention_mask=new_attention_mask,
                            use_cache=False
                        )
                        new_hidden_states = new_hidden_states[:, -1:, :]
                        initial_input_ids = updated_input_ids
                    
                    # 确保hidden state的形状正确 [1, 1, hidden_size]
                    if new_hidden_states.shape[1] != 1:
                        new_hidden_states = new_hidden_states[:, -1:, :]
                    
                    print(f"[Stage {stage_idx}] Generated hidden state for token {token_id} (shape: {new_hidden_states.shape})", file=sys.stderr, flush=True)
                    
                    seq_length += 1
                    
                    # 发送新的hidden state（单个token，已通过stage 0的layers）
                    new_tensor_dict = {"hidden_states": new_hidden_states}
                    communicator.send_tensor_dict(new_tensor_dict)
                    logger.info(f"Stage {stage_idx} sent new hidden state for token {token_id} (step {generation_step})")
                    
            except Exception as e:
                if "Empty" not in str(e):  # 忽略队列超时
                    logger.error(f"Stage {stage_idx} error: {e}", exc_info=True)
            
            communicator.close()
            
        elif is_last:
            # 最后阶段：从上一个stage接收，输出到output_queue
            print(f"[Stage {stage_idx}] ZeroMQ communicator initialized", file=sys.stderr, flush=True)
            logger.info(f"Stage {stage_idx} ZeroMQ communicator initialized")
            
            # 从stage 0加载tokenizer（因为只有stage 0有tokenizer文件）
            stage0_dir = Path(pipeline_dir) / "stage_0"
            tokenizer = AutoTokenizer.from_pretrained(str(stage0_dir), trust_remote_code=True)
            
            # 等待初始输入并生成多个token
            try:
                print(f"[Stage {stage_idx}] Waiting to receive initial hidden states...", file=sys.stderr, flush=True)
                logger.info(f"Stage {stage_idx} waiting to receive initial hidden states")
                
                # 使用ZeroMQ通信器接收tensor dict
                tensor_dict = communicator.recv_tensor_dict()
                if tensor_dict is None:
                    raise RuntimeError("Failed to receive tensor dict")
                
                hidden_states = tensor_dict["hidden_states"]
                print(f"[Stage {stage_idx}] Received initial hidden states (shape: {hidden_states.shape})", file=sys.stderr, flush=True)
                logger.info(f"Stage {stage_idx} received initial hidden states")
                
                # 生成token序列
                generated_tokens = []
                
                # 处理初始hidden states（来自prompt）
                print(f"[Stage {stage_idx}] Processing initial hidden states (shape: {hidden_states.shape})...", file=sys.stderr, flush=True)
                forward_result = stage.forward(hidden_states, use_cache=False)
                # 处理返回值（可能是tuple或单个tensor）
                if isinstance(forward_result, tuple):
                    logits = forward_result[0]
                else:
                    logits = forward_result
                
                # 解码 - 确保正确提取logits（取最后一个token的logits）
                if hasattr(logits, "shape"):
                    if len(logits.shape) == 3:
                        # Shape: [batch_size, seq_len, vocab_size]
                        next_token_logits = logits[0, -1, :]  # 取最后一个token的logits
                    elif len(logits.shape) == 2:
                        # Shape: [seq_len, vocab_size] or [batch_size, vocab_size]
                        if logits.shape[0] > logits.shape[1]:
                            # 可能是 [seq_len, vocab_size]
                            next_token_logits = logits[-1, :]
                        else:
                            # 可能是 [batch_size, vocab_size]
                            next_token_logits = logits[0, :] if logits.shape[0] == 1 else logits[-1, :]
                    else:
                        # 1D or other shape
                        next_token_logits = logits.flatten()
                else:
                    next_token_logits = logits
                
                # 获取第一个生成的token ID
                next_token_id = torch.argmax(next_token_logits, dim=-1)
                if isinstance(next_token_id, torch.Tensor):
                    next_token_id = next_token_id.item()
                
                generated_tokens.append(next_token_id)
                print(f"[Stage {stage_idx}] Generated first token: {next_token_id}", file=sys.stderr, flush=True)
                
                # 检查是否遇到EOS token
                if tokenizer.eos_token_id is not None and next_token_id == tokenizer.eos_token_id:
                    print(f"[Stage {stage_idx}] Generated EOS token at first step, stopping generation", file=sys.stderr, flush=True)
                else:
                    # 自回归生成循环：每个新token都需要通过整个pipeline
                    print(f"[Stage {stage_idx}] Starting autoregressive generation loop (max_new_tokens={max_new_tokens})...", file=sys.stderr, flush=True)
                    
                    for step in range(1, max_new_tokens):
                        # 将token ID发送回stage 0，stage 0会通过embedding得到新的hidden state
                        # 然后通过所有stages，最后stage会再次生成下一个token
                        communicator.send_token_id(next_token_id)
                        logger.debug(f"Stage {stage_idx} sent token ID {next_token_id} back to stage 0 (step {step})")
                        
                        # 等待接收新的hidden states（从stage 0通过所有stages传递过来）
                        # 注意：这里接收的是单个token的hidden state，shape应该是 [1, 1, hidden_size]
                        try:
                            new_tensor_dict = communicator.recv_tensor_dict()
                            if new_tensor_dict is None:
                                logger.error(f"Stage {stage_idx} failed to receive new hidden states at step {step}")
                                break
                            
                            new_hidden_states = new_tensor_dict["hidden_states"]
                            print(f"[Stage {stage_idx}] Received new hidden states for step {step+1} (shape: {new_hidden_states.shape})", file=sys.stderr, flush=True)
                            logger.debug(f"Stage {stage_idx} received new hidden states for token {step+1} (shape: {new_hidden_states.shape})")
                            
                            # Forward：处理单个token的hidden state（最后stage不使用KV cache）
                            forward_result = stage.forward(new_hidden_states, use_cache=False)
                            # 处理返回值（可能是tuple或单个tensor）
                            if isinstance(forward_result, tuple):
                                logits = forward_result[0]
                            else:
                                logits = forward_result
                            
                            # 解码 - 确保正确提取logits
                            if hasattr(logits, "shape"):
                                if len(logits.shape) == 3:
                                    # Shape: [batch_size, seq_len, vocab_size]
                                    next_token_logits = logits[0, -1, :]  # 取最后一个token的logits
                                elif len(logits.shape) == 2:
                                    # Shape: [seq_len, vocab_size] or [batch_size, vocab_size]
                                    if logits.shape[0] > logits.shape[1]:
                                        # 可能是 [seq_len, vocab_size]
                                        next_token_logits = logits[-1, :]
                                    else:
                                        # 可能是 [batch_size, vocab_size]
                                        next_token_logits = logits[0, :] if logits.shape[0] == 1 else logits[-1, :]
                                else:
                                    # 1D or other shape
                                    next_token_logits = logits.flatten()
                            else:
                                next_token_logits = logits
                            
                            # 获取下一个token ID
                            next_token_id = torch.argmax(next_token_logits, dim=-1)
                            if isinstance(next_token_id, torch.Tensor):
                                next_token_id = next_token_id.item()
                            
                            generated_tokens.append(next_token_id)
                            print(f"[Stage {stage_idx}] Generated token {step+1}: {next_token_id}", file=sys.stderr, flush=True)
                            
                            # 检查是否遇到EOS token
                            if tokenizer.eos_token_id is not None and next_token_id == tokenizer.eos_token_id:
                                print(f"[Stage {stage_idx}] Generated EOS token at step {step+1}, stopping generation", file=sys.stderr, flush=True)
                                break
                            
                            if (step + 1) % 10 == 0:
                                print(f"[Stage {stage_idx}] Generated {step+1}/{max_new_tokens} tokens", file=sys.stderr, flush=True)
                                
                        except Exception as e:
                            logger.error(f"Stage {stage_idx} error in autoregressive loop at step {step}: {e}", exc_info=True)
                            import traceback
                            traceback.print_exc()
                            break
                
                # 解码所有生成的tokens
                output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                print(f"[Stage {stage_idx}] Generated {len(generated_tokens)} tokens: '{output_text[:100]}...'", file=sys.stderr, flush=True)
                
                logger.info(f"Stage {stage_idx} generated {len(generated_tokens)} tokens")
                output_queue.put(output_text)
                
            except Exception as e:
                logger.error(f"Stage {stage_idx} error: {e}", exc_info=True)
                import traceback
                traceback.print_exc()
                output_queue.put(f"ERROR: {e}")
            
            communicator.close()
            
        else:
            # 中间阶段：从上一个stage接收，发送到下一个stage
            print(f"[Stage {stage_idx}] ZeroMQ communicator initialized", file=sys.stderr, flush=True)
            logger.info(f"Stage {stage_idx} ZeroMQ communicator initialized")
            
            # 处理初始输入和后续的自回归生成
            try:
                # 处理初始输入
                print(f"[Stage {stage_idx}] Waiting to receive initial hidden states...", file=sys.stderr, flush=True)
                logger.info(f"Stage {stage_idx} waiting to receive initial hidden states")
                
                tensor_dict = communicator.recv_tensor_dict()
                if tensor_dict is None:
                    raise RuntimeError("Failed to receive initial tensor dict")
                
                hidden_states = tensor_dict["hidden_states"]
                print(f"[Stage {stage_idx}] Received initial hidden states (shape: {hidden_states.shape})", file=sys.stderr, flush=True)
                logger.info(f"Stage {stage_idx} received initial hidden states")
                
                # Forward（中间stages不使用KV cache）
                forward_result = stage.forward(hidden_states, use_cache=False)
                # 处理返回值（可能是tuple或单个tensor）
                if isinstance(forward_result, tuple):
                    hidden_states = forward_result[0]
                else:
                    hidden_states = forward_result
                
                # 发送到下一个stage
                output_tensor_dict = {"hidden_states": hidden_states}
                communicator.send_tensor_dict(output_tensor_dict)
                logger.info(f"Stage {stage_idx} sent initial hidden states to next stage")
                
                # 处理自回归生成：循环接收和转发新的hidden states
                # 注意：中间stage需要持续处理，直到生成完成
                generation_step = 0
                while True:
                    try:
                        logger.debug(f"Stage {stage_idx} waiting for hidden states (generation step {generation_step})...")
                        tensor_dict = communicator.recv_tensor_dict()
                        if tensor_dict is None:
                            # 超时或错误，退出循环
                            logger.info(f"Stage {stage_idx} no more hidden states to process (timeout or error, step {generation_step})")
                            break
                        
                        hidden_states = tensor_dict["hidden_states"]
                        print(f"[Stage {stage_idx}] Received hidden states for step {generation_step} (shape: {hidden_states.shape})", file=sys.stderr, flush=True)
                        logger.debug(f"Stage {stage_idx} received hidden states (shape: {hidden_states.shape}, step {generation_step})")
                        
                        # Forward：处理hidden states（中间stages不使用KV cache）
                        forward_result = stage.forward(hidden_states, use_cache=False)
                        # 处理返回值（可能是tuple或单个tensor）
                        if isinstance(forward_result, tuple):
                            hidden_states = forward_result[0]
                        else:
                            hidden_states = forward_result
                        
                        # 确保输出形状正确（如果是单个token输入，输出也应该是单个token）
                        if tensor_dict["hidden_states"].shape[1] == 1 and hidden_states.shape[1] != 1:
                            # 如果输入是单个token但输出不是，取最后一个
                            hidden_states = hidden_states[:, -1:, :]
                        
                        # 发送到下一个stage
                        output_tensor_dict = {"hidden_states": hidden_states}
                        communicator.send_tensor_dict(output_tensor_dict)
                        print(f"[Stage {stage_idx}] Sent hidden states to next stage (step {generation_step}, shape: {hidden_states.shape})", file=sys.stderr, flush=True)
                        logger.debug(f"Stage {stage_idx} sent hidden states to next stage (step {generation_step})")
                        
                        generation_step += 1
                    except Exception as e:
                        # 检查是否是ZeroMQ超时错误
                        error_str = str(e)
                        if "timeout" in error_str.lower() or "Again" in error_str:
                            logger.info(f"Stage {stage_idx} receive timeout, assuming generation complete (step {generation_step})")
                            break
                        else:
                            logger.error(f"Stage {stage_idx} error in generation loop (step {generation_step}): {e}", exc_info=True)
                            import traceback
                            traceback.print_exc()
                            break
                
            except Exception as e:
                logger.error(f"Stage {stage_idx} error: {e}", exc_info=True)
                import traceback
                traceback.print_exc()
            
            communicator.close()
            
    except Exception as e:
        print(f"[Stage {stage_idx}] FATAL ERROR: {e}", file=sys.stderr, flush=True)
        logger.error(f"Stage {stage_idx} fatal error: {e}", exc_info=True)
        if output_queue is not None:
            try:
                output_queue.put(f"ERROR: Stage {stage_idx} failed: {e}")
            except:
                pass
    finally:
        print(f"[Stage {stage_idx}] Worker process exiting", file=sys.stderr, flush=True)
        logger.info(f"Stage {stage_idx} worker process exiting")


def test_pipeline_multiprocess(
    pipeline_dir: str,
    num_stages: int,
    test_input: str,
    zmq_ports: Optional[list[int]] = None,
    device: str = "cpu",
    max_new_tokens: int = 256,
):
    """多进程测试pipeline推理"""
    # 设置multiprocessing启动方法为'spawn'（CUDA需要）
    if device.startswith("cuda"):
        try:
            multiprocessing.set_start_method('spawn', force=True)
            logger.info("Set multiprocessing start method to 'spawn' for CUDA support")
        except RuntimeError:
            # 如果已经设置过，忽略错误
            pass
    
    pipeline_dir = Path(pipeline_dir)

    logger.info(f"Testing pipeline from {pipeline_dir} (multiprocess mode)")
    logger.info(f"Number of stages: {num_stages}")
    logger.info(f"Test input: {test_input}")

    # 设置ZeroMQ端口
    if zmq_ports is None:
        base_port = 5555
        zmq_ports = [base_port + i for i in range(num_stages - 1)]
    
    # 创建进程间通信队列（只有第一个和最后一个stage需要）
    input_queue = multiprocessing.Queue() if num_stages > 0 else None
    output_queue = multiprocessing.Queue() if num_stages > 0 else None
    
    # 启动所有stage进程（并行加载）
    processes = []
    logger.info("Starting all stage processes in parallel...")
    logger.info(f"Max new tokens: {max_new_tokens}")
    for i in range(num_stages):
        p = multiprocessing.Process(
            target=stage_worker_process,
            args=(
                i,
                str(pipeline_dir),
                num_stages,
                zmq_ports,
                device,
                input_queue if i == 0 else None,
                output_queue if i == num_stages - 1 else None,
                max_new_tokens,
            ),
        )
        p.start()
        processes.append(p)
        logger.info(f"Started stage {i} process (PID: {p.pid})")
    
    # 等待所有进程加载完成（给一些时间让模型加载）
    logger.info("Waiting for all stages to load models...")
    print("Waiting for all stages to load models...", flush=True)
    max_wait_time = 120  # 最多等待120秒（模型加载可能需要更长时间）
    start_time = time.time()
    for i in range(max_wait_time):
        time.sleep(2)  # 每2秒检查一次
        alive_count = sum(1 for p in processes if p.is_alive())
        elapsed = time.time() - start_time
        if alive_count == num_stages:
            if i % 5 == 0:  # 每10秒打印一次
                logger.info(f"All {alive_count} processes are alive, waiting for models to load... ({elapsed:.1f}s)")
                print(f"All {alive_count} processes are alive, waiting for models to load... ({elapsed:.1f}s)", flush=True)
        else:
            dead_pids = [p.pid for p in processes if not p.is_alive()]
            logger.warning(f"Only {alive_count}/{num_stages} processes are alive. Dead PIDs: {dead_pids}")
            print(f"WARNING: Only {alive_count}/{num_stages} processes are alive. Dead PIDs: {dead_pids}", flush=True)
            # 检查是否有进程异常退出
            for p in processes:
                if not p.is_alive() and p.exitcode != 0:
                    logger.error(f"Process {p.pid} exited with code {p.exitcode}")
                    print(f"ERROR: Process {p.pid} exited with code {p.exitcode}", flush=True)
    
    final_alive = sum(1 for p in processes if p.is_alive())
    if final_alive < num_stages:
        logger.error(f"Only {final_alive}/{num_stages} processes survived the loading phase")
        print(f"ERROR: Only {final_alive}/{num_stages} processes survived the loading phase", flush=True)
        raise RuntimeError(f"Only {final_alive}/{num_stages} processes survived the loading phase")
    
    logger.info("All processes started. Sending input to stage 0...")
    print("All processes started. Sending input to stage 0...", flush=True)
    
    try:
        # 发送输入到第一个stage
        print(f"Sending input to stage 0: {test_input}", flush=True)
        input_queue.put(test_input)
        logger.info("Input sent to stage 0, waiting for output from last stage...")
        print("Input sent to stage 0, waiting for output from last stage...", flush=True)
        
        # 等待最后一个stage的输出
        try:
            output = output_queue.get(timeout=300)  # 增加超时时间到5分钟
            logger.info(f"Inference output: {output}")
            print(f"Inference output: {output}", flush=True)
            return output
        except Exception as queue_error:
            logger.error(f"Failed to get output from queue: {queue_error}", exc_info=True)
            print(f"ERROR: Failed to get output from queue: {queue_error}", flush=True)
            # 检查进程状态
            for i, p in enumerate(processes):
                if not p.is_alive():
                    logger.error(f"Process {i} (PID: {p.pid}) is not alive, exitcode: {p.exitcode}")
                    print(f"ERROR: Process {i} (PID: {p.pid}) is not alive, exitcode: {p.exitcode}", flush=True)
            raise
        
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        print(f"ERROR: Inference failed: {e}", flush=True)
        raise
    finally:
        # 发送结束信号
        if num_stages > 0:
            input_queue.put(None)
        
        # 等待所有进程结束
        for p in processes:
            p.join(timeout=10)
            if p.is_alive():
                logger.warning(f"Process {p.pid} did not terminate, killing...")
                p.terminate()
                p.join()


def test_pipeline(
    pipeline_dir: str,
    num_stages: int,
    test_input: str,
    zmq_ports: Optional[list[int]] = None,
    device: str = "cpu",
    use_multiprocess: bool = True,
    max_new_tokens: int = 256,
):
    """测试pipeline推理"""
    if use_multiprocess:
        return test_pipeline_multiprocess(
            pipeline_dir=pipeline_dir,
            num_stages=num_stages,
            test_input=test_input,
            zmq_ports=zmq_ports,
            device=device,
            max_new_tokens=max_new_tokens,
        )
    else:
        # 单进程模式（原有实现）
        pipeline_dir = Path(pipeline_dir)

        logger.info(f"Testing pipeline from {pipeline_dir}")
        logger.info(f"Number of stages: {num_stages}")
        logger.info(f"Test input: {test_input}")

        # 创建pipeline
        pipeline = ZeroMQPipeline(
            pipeline_dir=pipeline_dir,
            num_stages=num_stages,
            zmq_ports=zmq_ports,
            device=device,
        )

        # 运行推理
        try:
            output = pipeline.inference(test_input)
            logger.info(f"Inference output: {output}")
            return output
        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Test offline pipeline with ZeroMQ"
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
        "--test-input",
        type=str,
        default="Hello, world!",
        help="Test input text",
    )
    parser.add_argument(
        "--zmq-ports",
        type=str,
        default=None,
        help="Comma-separated list of ZeroMQ ports (default: auto)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on (default: cpu)",
    )
    parser.add_argument(
        "--multiprocess",
        action="store_true",
        default=True,
        help="Use multiprocess mode (default: True)",
    )
    parser.add_argument(
        "--single-process",
        action="store_true",
        help="Use single process mode instead of multiprocess",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate (default: 256)",
    )

    args = parser.parse_args()

    # 解析端口
    zmq_ports = None
    if args.zmq_ports:
        zmq_ports = [int(p) for p in args.zmq_ports.split(",")]

    use_multiprocess = args.multiprocess and not args.single_process

    print(f"Starting pipeline test:", flush=True)
    print(f"  Pipeline dir: {args.pipeline_dir}", flush=True)
    print(f"  Num stages: {args.num_stages}", flush=True)
    print(f"  Test input: {args.test_input}", flush=True)
    print(f"  Device: {args.device}", flush=True)
    print(f"  Multiprocess: {use_multiprocess}", flush=True)

    try:
        output = test_pipeline(
            pipeline_dir=args.pipeline_dir,
            num_stages=args.num_stages,
            test_input=args.test_input,
            zmq_ports=zmq_ports,
            device=args.device,
            use_multiprocess=use_multiprocess,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"\n=== FINAL OUTPUT ===", flush=True)
        print(f"{output}", flush=True)
        print(f"===================", flush=True)
    except Exception as e:
        print(f"\n=== ERROR ===", flush=True)
        print(f"{e}", flush=True)
        print(f"=============", flush=True)
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

