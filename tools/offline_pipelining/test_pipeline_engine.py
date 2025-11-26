#!/usr/bin/env python3
"""
Optimized vLLM Engine-based Pipeline Test with ZeroMQ
Fixed issues: position encoding, KV cache management, attention context
"""
import argparse
import multiprocessing
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Union, Any, Dict, List
import torch
from vllm.engine.arg_utils import EngineArgs
from vllm.distributed.parallel_state import get_pp_group, initialize_model_parallel, init_distributed_environment
from vllm.entrypoints.llm import LLM
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.sequence import IntermediateTensors
from vllm.model_executor.models.utils import PPMissingLayer
from transformers import AutoTokenizer
# Import zmq_communicator
try:
    from .zmq_communicator import ZeroMQCommunicator
except ImportError:
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    from zmq_communicator import ZeroMQCommunicator
logger = init_logger(__name__)
# Global ZeroMQ communicator for each stage process
_zmq_communicator: Optional[ZeroMQCommunicator] = None




class ZeroMQPPGroup:
    """Mock PP Group that uses ZeroMQ instead of PyTorch distributed"""
    
    def __init__(self, stage_idx: int, num_stages: int, communicator: ZeroMQCommunicator):
        self.stage_idx = stage_idx
        self.num_stages = num_stages
        self.communicator = communicator
        self.rank = stage_idx
        self.world_size = num_stages
        self.rank_in_group = stage_idx
        self.is_first_rank = (stage_idx == 0)
        self.is_last_rank = (stage_idx == num_stages - 1)
        self.first_rank = 0
        self.last_rank = num_stages - 1
    
    def send_tensor_dict(
        self,
        tensor_dict: dict[str, Union[torch.Tensor, Any]],
        dst: Optional[int] = None,
        all_gather_group=None,
        all_gather_tensors: Optional[dict[str, bool]] = None,
    ) -> Optional[dict[str, Union[torch.Tensor, Any]]]:
        """Send tensor dict via ZeroMQ"""
        if self.is_last_rank:
            raise RuntimeError("Last stage cannot send tensor dict")
        
        # Use ZeroMQ communicator to send
        self.communicator.send_tensor_dict(tensor_dict, dst)
        return None
    
    def recv_tensor_dict(
        self,
        src: Optional[int] = None,
        all_gather_group=None,
        all_gather_tensors: Optional[dict[str, bool]] = None,
    ) -> Optional[dict[str, Union[torch.Tensor, Any]]]:
        """Receive tensor dict via ZeroMQ"""
        if self.is_first_rank:
            raise RuntimeError("First stage cannot receive tensor dict")
        
        # Use ZeroMQ communicator to receive
        return self.communicator.recv_tensor_dict(src)
    
    def send_object(self, obj, dst: int | None = None):
        """Not used in ZeroMQ mode"""
        pass
    
    def recv_object(self, src: int | None = None):
        """Not used in ZeroMQ mode"""
        return None


# ----------------- Pipeline State Manager -----------------
class PipelineStateManager:
    """Manages global state across pipeline stages"""
    
    def __init__(self, stage_idx: int, num_stages: int):
        self.stage_idx = stage_idx
        self.num_stages = num_stages
        self.global_seq_len = 0  # Length of the full sequence across all stages
        self.kv_caches = {}  # {layer_id: (key_cache, value_cache)}
        self.token_positions = []  # Global token positions for all tokens
        
    def update_from_tensor_dict(self, tensor_dict: Dict[str, Any]):
        """Update state from received tensor dict"""
        if 'global_seq_len' in tensor_dict:
            self.global_seq_len = tensor_dict['global_seq_len']
        if 'token_positions' in tensor_dict:
            self.token_positions = tensor_dict['token_positions'].tolist()
        # KV cache updates handled separately in attention layers
    
    def prepare_tensor_dict(self, hidden_states, residual, is_decode: bool = False):
        """Prepare tensor dict with necessary state information"""
        tensor_dict = {
            "hidden_states": hidden_states,
            "residual": residual,
            "global_seq_len": self.global_seq_len,
            "is_decode": is_decode,
        }
        if self.token_positions:
            tensor_dict["token_positions"] = torch.tensor(self.token_positions, device=hidden_states.device)
        return tensor_dict
    
    def update_positions(self, current_positions: torch.Tensor, is_initial_prompt: bool = False):
        """Update global token positions based on current stage's processing"""
        if is_initial_prompt:
            # For initial prompt, set token positions from 0 to seq_len-1
            self.global_seq_len = current_positions.shape[-1]
            self.token_positions = current_positions.tolist()
        else:
            # For decode, append new position
            new_position = self.global_seq_len
            self.token_positions.append(new_position)
            self.global_seq_len += 1
    
    def get_current_position(self) -> int:
        """Get current global position for next token"""
        return self.global_seq_len

# ----------------- Improved attention context handling -----------------
def _register_stage_attention_layers(
    model,
    vllm_config,
    stage_start_layer: int,
    stage_end_layer: int | None = None,
    global_layer_map: Dict[int, int] | None = None,
    pipeline_state: PipelineStateManager = None,
    stage_idx: int | None = None
) -> list[str]:
    """Register attention layers into static_forward_context with proper global indexing."""
    forward_context = vllm_config.compilation_config.static_forward_context
    # Get stage_idx from pipeline_state if available, otherwise use stage_idx parameter
    actual_stage_idx = pipeline_state.stage_idx if pipeline_state is not None else (stage_idx if stage_idx is not None else 0)
    logger.debug(f"[Stage {actual_stage_idx}] Registering attention layers: start={stage_start_layer}, end={stage_end_layer}")
    
    # Clear stale keys
    try:
        stale_keys = [k for k in forward_context.keys() if k.startswith("model.layers.")]
    except Exception:
        if hasattr(forward_context, "no_compile_layers"):
            stale_keys = [k for k in forward_context.no_compile_layers.keys() if k.startswith("model.layers.")]
        else:
            stale_keys = []
    for k in stale_keys:
        try:
            forward_context.pop(k, None)
        except Exception:
            try:
                if hasattr(forward_context, "no_compile_layers"):
                    forward_context.no_compile_layers.pop(k, None)
            except Exception:
                pass

    # Get base model (unwrap if necessary)
    base_model = model.model if hasattr(model, "model") else model
    layers = getattr(base_model, "layers", None)
    if layers is None:
        logger.warning("[DEBUG] No layers found in model")
        return []

    registered_names = []
    attention_layers = {}
    
    # Get all relevant layer numbers for this stage
    layer_indices = range(len(layers))
    if stage_end_layer is not None:
        layer_indices = [i for i in layer_indices if i < (stage_end_layer - stage_start_layer)]
    
    for local_idx in layer_indices:
        layer_module = layers[local_idx]
        
        # Skip missing layers
        if isinstance(layer_module, PPMissingLayer):
            continue
            
        # Determine global layer index
        if global_layer_map is not None and local_idx in global_layer_map:
            global_idx = global_layer_map[local_idx]
        else:
            # Fallback to local index + stage start
            global_idx = stage_start_layer + local_idx
        
        # Get attention module
        attn_module = getattr(layer_module, "self_attn", None)
        if attn_module is None:
            continue
            
        attention_layer = getattr(attn_module, "attn", attn_module)
        
        # Build proper layer name with global index
        layer_name = f"model.layers.{global_idx}.self_attn.attn"
        
        # Register in forward context
        try:
            forward_context[layer_name] = attention_layer
        except Exception:
            if hasattr(forward_context, "no_compile_layers"):
                try:
                    forward_context.no_compile_layers[layer_name] = attention_layer
                except Exception as e:
                    logger.warning(f"Failed to set forward_context entry for {layer_name}: {e}")
            else:
                logger.warning(f"forward_context does not accept item assignment; skipping {layer_name}")
        
        # Also store in attention_layers dict for easier access later
        attention_layers[global_idx] = attention_layer
        registered_names.append(layer_name)
        logger.debug(f"[Stage {actual_stage_idx}] REGISTER ATTENTION: {layer_name} (local_idx={local_idx}, global_idx={global_idx})")
    
    logger.info(f"[Stage {actual_stage_idx}] Registered {len(registered_names)} attention layers")
    return registered_names, attention_layers

def stage_worker_process(
    stage_idx: int,
    pipeline_dir: str,
    num_stages: int,
    zmq_ports: list[int],
    device: str,
    dist_init_path: str,
    input_queue: Optional[multiprocessing.Queue],
    output_queue: Optional[multiprocessing.Queue],
    max_new_tokens: int = 256,
):
    """Optimized stage worker process using vLLM engine with ZeroMQ communication"""
    import os
    try:
        # Set device for this stage
        if device.startswith("cuda") and torch.cuda.device_count() > 0:
            gpu_id = stage_idx % torch.cuda.device_count()
            stage_device = f"cuda:{gpu_id}"
            torch.cuda.set_device(torch.device(stage_device))
            logger.info(f"[Stage {stage_idx}] Using GPU device: {stage_device}")
        else:
            stage_device = device
            logger.info(f"[Stage {stage_idx}] Using CPU device")
        
        def log_stage(message: str, level="INFO") -> None:
            formatted = f"[Stage {stage_idx}] {message}"
            if level == "DEBUG":
                logger.debug(formatted)
            elif level == "WARNING":
                logger.warning(formatted)
            elif level == "ERROR":
                logger.error(formatted)
            else:
                logger.info(formatted)
        
        log_stage(f"Worker process started on {stage_device} (PID: {os.getpid()})")
        
        # Initialize pipeline state manager
        pipeline_state = PipelineStateManager(stage_idx, num_stages)
        log_stage("Initialized pipeline state manager")
        
        # Setup ZeroMQ communicator
        communicator = ZeroMQCommunicator(
            stage_idx=stage_idx,
            num_stages=num_stages,
            zmq_ports=zmq_ports,
            device=stage_device,
            timeout_ms=300000,  # 5 minutes timeout
        )
        log_stage("ZeroMQ communicator initialized")
        
        # Setup ZeroMQ PP group
        setup_zeromq_pp_group(stage_idx, num_stages, communicator)
        log_stage("ZeroMQ PP group patched")
        
        # Load stage model
        stage_dir = Path(pipeline_dir) / f"stage_{stage_idx}"
        if not stage_dir.exists():
            raise FileNotFoundError(f"Stage directory not found: {stage_dir}")
        log_stage(f"Loading model from {stage_dir}")
        
        # Load config
        import json
        config_path = stage_dir / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            stage_config = json.load(f)
        
        pipeline_info = stage_config.get("_pipeline_info", {})
        stage_start_layer = pipeline_info.get("start_layer")
        stage_end_layer = pipeline_info.get("end_layer")
        global_layer_map = None
        
        if "global_layer_map" in pipeline_info:
            global_layer_map_serializable = pipeline_info["global_layer_map"]
            global_layer_map = {
                int(local_idx): global_idx
                for local_idx, global_idx in global_layer_map_serializable.items()
            }
            log_stage(f"Loaded global_layer_map with {len(global_layer_map)} entries")
        
        # Determine model dtype
        dtype_str = stage_config.get("dtype", stage_config.get("torch_dtype", "float16"))
        model_dtype = "float16" if "float16" in str(dtype_str) else "bfloat16" if "bfloat16" in str(dtype_str) else "float16"
        
        # Initialize model components
        from vllm.config import ModelConfig, VllmConfig, LoadConfig, DeviceConfig, CompilationConfig, CompilationMode, ParallelConfig
        from vllm.model_executor.model_loader.utils import initialize_model
        from vllm.config import set_current_vllm_config
        from vllm.forward_context import set_forward_context, get_forward_context
        
        # Create model config
        model_config = ModelConfig(
            model=str(stage_dir),
            dtype=model_dtype,
            trust_remote_code=True,
            task=None,
            tokenizer=str(stage_dir),
            tokenizer_mode="auto",
            seed=0,
            quantization=None,
        )
        
        # Get original number of layers
        original_num_layers = pipeline_info.get("original_num_hidden_layers", model_config.hf_config.num_hidden_layers)
        
        # Create VllmConfig
        load_config = LoadConfig(load_format="auto", download_dir=None)
        device_for_config = stage_device if stage_device != "cpu" else "cpu"
        device_config = DeviceConfig(device=device_for_config)
        compilation_config = CompilationConfig(mode=CompilationMode.NONE)
        parallel_config = ParallelConfig(
            pipeline_parallel_size=num_stages,
            tensor_parallel_size=1,
        )
        vllm_config = VllmConfig(
            model_config=model_config,
            load_config=load_config,
            device_config=device_config,
            compilation_config=compilation_config,
            parallel_config=parallel_config,
        )
        
        log_stage(f"Initializing model on device: {stage_device}")
        
        # Initialize model
        with set_current_vllm_config(vllm_config=vllm_config):
            model = initialize_model(
                vllm_config=vllm_config,
                model_config=model_config,
            )
        
        # Load weights directly to target device
        weights_file = None
        if (stage_dir / "model.safetensors").exists():
            weights_file = str(stage_dir / "model.safetensors")
        elif (stage_dir / "pytorch_model.bin").exists():
            weights_file = str(stage_dir / "pytorch_model.bin")
        
        if weights_file:
            log_stage(f"Loading weights from {weights_file} to device {stage_device}")
            target_device = stage_device if stage_device != "cpu" else "cpu"
            
            if weights_file.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(weights_file, device=target_device)
            else:
                state_dict = torch.load(weights_file, map_location=target_device, weights_only=True)
            
            # Convert all weights to target dtype
            log_stage(f"Converting weights to {model_dtype}...")
            target_dtype = torch.float16 if model_dtype == "float16" else torch.bfloat16
            for key, value in state_dict.items():
                if isinstance(value, torch.Tensor) and value.dtype != target_dtype:
                    state_dict[key] = value.to(dtype=target_dtype)
            
            # Load weights
            from vllm.model_executor.models.utils import AutoWeightsLoader
            weights_loader = AutoWeightsLoader(model)
            weights = [(k, v) for k, v in state_dict.items()]
            weights_loader.load_weights(weights)
            log_stage("Weights loaded successfully")
        
        # Process weights after loading
        from vllm.model_executor.model_loader.utils import process_weights_after_loading
        target_torch_device = torch.device(stage_device if stage_device != "cpu" else "cpu")
        process_weights_after_loading(
            model,
            model_config,
            target_torch_device
        )
        
        # Ensure model is on correct device
        if stage_device != "cpu":
            model = model.to(stage_device)
        
        # Ensure all parameters are in target dtype
        target_dtype = torch.float16 if model_dtype == "float16" else torch.bfloat16
        for param in model.parameters():
            if param.dtype != target_dtype:
                param.data = param.data.to(dtype=target_dtype)
        
        model.eval()
        log_stage(f"Model initialized on {next(model.parameters()).device} with dtype {target_dtype}")
        
        # Register attention layers for this stage
        registered_names, attention_layers = _register_stage_attention_layers(
            model=model,
            vllm_config=vllm_config,
            stage_start_layer=stage_start_layer or 0,
            stage_end_layer=stage_end_layer,
            global_layer_map=global_layer_map,
            pipeline_state=pipeline_state
        )
        
        # Initialize forward context if not already set
        try:
            ctx = get_forward_context()
            log_stage("Forward context already set")
        except (AssertionError, RuntimeError):
            try:
                set_forward_context(None, vllm_config, num_tokens=1)
                log_stage("Initialized default forward context")
            except Exception as ex:
                log_stage(f"Failed to initialize forward context: {ex}", "WARNING")
        
        # Load tokenizer (only first stage)
        tokenizer = None
        if stage_idx == 0:
            tokenizer = AutoTokenizer.from_pretrained(
                str(stage_dir), trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            log_stage("Tokenizer loaded")
        
        # Get model weight dtype for dtype consistency
        def get_model_weight_dtype(model):
            if hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > 0:
                first_layer = model.model.layers[0]
                if hasattr(first_layer, "self_attn") and hasattr(first_layer.self_attn, "q_proj"):
                    return first_layer.self_attn.q_proj.weight.dtype
                elif hasattr(first_layer, "mlp") and hasattr(first_layer.mlp, "gate_proj"):
                    return first_layer.mlp.gate_proj.weight.dtype
            return next(model.parameters()).dtype
        
        model_weight_dtype = get_model_weight_dtype(model)
        log_stage(f"Model weight dtype: {model_weight_dtype}")
        
        # Patch forward method for pipeline compatibility
        if hasattr(model, "model") and hasattr(model.model, "forward"):
            original_forward = model.model.forward
            
            def patched_forward(
                self_model,
                input_ids=None,
                positions=None,
                intermediate_tensors=None,
                inputs_embeds=None,
                kv_caches=None,
                attn_metadata=None,
                **kwargs
            ):
                """Optimized forward pass that correctly handles pipeline stages"""
                # Determine if this is the first stage (receives input_ids)
                is_first_stage = (intermediate_tensors is None and (input_ids is not None or inputs_embeds is not None))
                
                # Handle input for first stage
                if is_first_stage:
                    if inputs_embeds is not None:
                        hidden_states = inputs_embeds
                    else:
                        hidden_states = self_model.embed_tokens(input_ids)
                    residual = None
                else:
                    # Middle or last stage - receive hidden states from previous stage
                    assert intermediate_tensors is not None
                    hidden_states = intermediate_tensors.tensors["hidden_states"]
                    residual = intermediate_tensors.tensors.get("residual", None)
                
                # Ensure correct dtype
                if hidden_states.dtype != model_weight_dtype:
                    hidden_states = hidden_states.to(dtype=model_weight_dtype)
                if residual is not None and residual.dtype != model_weight_dtype:
                    residual = residual.to(dtype=model_weight_dtype)
                
                # Process through layers
                aux_hidden_states = []
                for idx, layer in enumerate(self_model.layers):
                    if idx in self_model.aux_hidden_state_layers:
                        aux_hidden_states.append(hidden_states + residual if residual is not None else hidden_states)
                    
                    # Process layer with KV cache and attention metadata
                    hidden_states, residual = layer(
                        positions=positions,
                        hidden_states=hidden_states,
                        residual=residual,
                        kv_cache=None,  # KV cache handled by attention layers directly
                        attn_metadata=attn_metadata
                    )
                
                # Determine if this is the last stage
                is_last_stage = (stage_idx == num_stages - 1)
                
                if not is_last_stage:
                    # Return intermediate tensors for next stage
                    return IntermediateTensors({
                        "hidden_states": hidden_states,
                        "residual": residual
                    })
                
                # Last stage: apply norm and return final hidden states
                hidden_states, _ = self_model.norm(hidden_states, residual)
                
                if hidden_states.dim() == 3:
                    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
                
                if len(aux_hidden_states) > 0:
                    return hidden_states, aux_hidden_states
                
                return hidden_states
            
            # Apply the patch
            import types
            model.model.forward = types.MethodType(patched_forward, model.model)
            log_stage("Patched forward method for pipeline compatibility")
        
        # Pipeline execution based on stage position
        is_first_stage = (stage_idx == 0)
        is_last_stage = (stage_idx == num_stages - 1)
        
        try:
            if is_first_stage:
                # First stage: process input prompt and start generation
                log_stage("Waiting for input from queue...")
                input_text = input_queue.get(timeout=300)
                if input_text is None:
                    log_stage("Received exit signal")
                    return
                
                log_stage(f"Received input: {input_text}")
                
                # Tokenize input
                inputs = tokenizer(input_text, return_tensors="pt")
                input_ids = inputs["input_ids"].to(stage_device)
                batch_size, seq_length = input_ids.shape
                pipeline_state.global_seq_len = seq_length
                
                # Create position IDs for the prompt
                positions = torch.arange(seq_length, dtype=torch.long, device=stage_device)
                pipeline_state.update_positions(positions, is_initial_prompt=True)
                
                # Process the prompt through the model
                log_stage(f"Processing prompt of length {seq_length}...")
                with set_current_vllm_config(vllm_config=vllm_config):
                    with set_forward_context(None, vllm_config, num_tokens=seq_length):
                        output = model(
                            input_ids=input_ids.flatten(),
                            positions=positions,
                            intermediate_tensors=None,
                        )
                
                # Send output to next stage
                if isinstance(output, IntermediateTensors):
                    hidden_states = output["hidden_states"]
                    residual = output.get("residual")
                    
                    # Prepare tensor dict with state information
                    tensor_dict = pipeline_state.prepare_tensor_dict(
                        hidden_states, residual, is_decode=False
                    )
                    
                    if not is_last_stage:
                        communicator.send_tensor_dict(tensor_dict)
                        log_stage(f"Sent initial hidden states to next stage. Global seq len: {pipeline_state.global_seq_len}")
                
                # Start generation loop
                generated_tokens = []
                for step in range(max_new_tokens):
                    if step == 0 and not is_last_stage:
                        # Wait for first token from last stage
                        token_id = communicator.recv_token_id()
                        if token_id is None:
                            break
                    elif step > 0:
                        # Receive token from last stage
                        token_id = communicator.recv_token_id()
                        if token_id is None:
                            break
                    
                    if step > 0:
                        generated_tokens.append(token_id)
                    
                    # Prepare input for next token
                    new_token_tensor = torch.tensor([[token_id]], device=stage_device, dtype=torch.long)
                    current_position = torch.tensor([pipeline_state.get_current_position()], device=stage_device, dtype=torch.long)
                    
                    # Update pipeline state
                    pipeline_state.update_positions(current_position)
                    
                    # Process new token
                    with set_current_vllm_config(vllm_config=vllm_config):
                        with set_forward_context(None, vllm_config, num_tokens=1):
                            new_output = model(
                                input_ids=new_token_tensor.flatten(),
                                positions=current_position,
                                intermediate_tensors=None,
                            )
                    
                    # Send to next stage
                    if isinstance(new_output, IntermediateTensors):
                        hidden_states = new_output["hidden_states"]
                        residual = new_output.get("residual")
                        
                        # Prepare tensor dict with state information
                        tensor_dict = pipeline_state.prepare_tensor_dict(
                            hidden_states, residual, is_decode=True
                        )
                        
                        if not is_last_stage:
                            communicator.send_tensor_dict(tensor_dict)
                            log_stage(f"Sent hidden state for token {step+1} with global position {current_position.item()}")
                    
                    # Check for EOS token
                    if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                        log_stage("Generated EOS token, stopping generation")
                        break
                
                # Generation complete
                communicator.send_token_id(None)  # Signal end of generation
                log_stage(f"Generation complete. Generated {len(generated_tokens)} tokens")
            
            elif is_last_stage:
                # Last stage: process inputs from previous stage and generate tokens
                log_stage("Waiting for initial activations...")
                
                # Load tokenizer for decoding
                stage0_dir = Path(pipeline_dir) / "stage_0"
                tokenizer = AutoTokenizer.from_pretrained(str(stage0_dir), trust_remote_code=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Receive initial activations
                tensor_dict = communicator.recv_tensor_dict()
                if tensor_dict is None:
                    raise RuntimeError("Failed to receive initial tensor dict")
                
                # Update pipeline state
                pipeline_state.update_from_tensor_dict(tensor_dict)
                initial_seq_len = pipeline_state.global_seq_len
                
                log_stage(f"Received initial activations. Global seq len: {initial_seq_len}")
                
                # Process initial activations
                hidden_states = tensor_dict["hidden_states"]
                residual = tensor_dict.get("residual")
                
                # Ensure correct dtype
                if hidden_states.dtype != model_weight_dtype:
                    hidden_states = hidden_states.to(dtype=model_weight_dtype)
                if residual is not None and residual.dtype != model_weight_dtype:
                    residual = residual.to(dtype=model_weight_dtype)
                
                # Get positions from tensor dict or create default
                positions = tensor_dict.get("token_positions")
                if positions is None:
                    positions = torch.arange(initial_seq_len, dtype=torch.long, device=hidden_states.device)
                
                # Process through model
                intermediate_tensors = IntermediateTensors({
                    "hidden_states": hidden_states,
                    "residual": residual
                })
                
                with set_current_vllm_config(vllm_config=vllm_config):
                    with set_forward_context(None, vllm_config, num_tokens=initial_seq_len):
                        output = model(
                            input_ids=None,
                            positions=positions,
                            intermediate_tensors=intermediate_tensors,
                        )
                
                # Generate first token
                if hasattr(model, "compute_logits"):
                    logits = model.compute_logits(output)
                else:
                    logits = output
                
                # Get the last token's logits
                if logits.dim() == 3:
                    next_token_logits = logits[0, -1, :]
                elif logits.dim() == 2:
                    next_token_logits = logits[-1, :]
                else:
                    next_token_logits = logits
                
                next_token_id = torch.argmax(next_token_logits, dim=-1).item()
                generated_tokens = [next_token_id]
                log_stage(f"Generated first token: {next_token_id} (text: '{tokenizer.decode([next_token_id], skip_special_tokens=True)}')")
                
                # Send first token back to first stage
                communicator.send_token_id(next_token_id)
                
                # Generation loop
                for step in range(1, max_new_tokens):
                    # Receive new hidden states from previous stage
                    new_tensor_dict = communicator.recv_tensor_dict()
                    if new_tensor_dict is None:
                        log_stage("No more hidden states received, stopping generation")
                        break
                    
                    # Update pipeline state
                    pipeline_state.update_from_tensor_dict(new_tensor_dict)
                    
                    # Process new hidden states
                    hidden_states = new_tensor_dict["hidden_states"]
                    residual = new_tensor_dict.get("residual")
                    
                    # Ensure correct dtype
                    if hidden_states.dtype != model_weight_dtype:
                        hidden_states = hidden_states.to(dtype=model_weight_dtype)
                    if residual is not None and residual.dtype != model_weight_dtype:
                        residual = residual.to(dtype=model_weight_dtype)
                    
                    # Get position from tensor dict
                    current_position = new_tensor_dict.get("token_positions")
                    if current_position is None:
                        current_position = torch.tensor([pipeline_state.get_current_position()-1], device=hidden_states.device)
                    
                    # Process through model
                    new_intermediate_tensors = IntermediateTensors({
                        "hidden_states": hidden_states,
                        "residual": residual
                    })
                    
                    with set_current_vllm_config(vllm_config=vllm_config):
                        with set_forward_context(None, vllm_config, num_tokens=1):
                            new_output = model(
                                input_ids=None,
                                positions=current_position,
                                intermediate_tensors=new_intermediate_tensors,
                            )
                    
                    # Generate next token
                    if hasattr(model, "compute_logits"):
                        new_logits = model.compute_logits(new_output)
                    else:
                        new_logits = new_output
                    
                    # Get the last token's logits
                    if new_logits.dim() == 3:
                        next_token_logits = new_logits[0, -1, :]
                    elif new_logits.dim() == 2:
                        next_token_logits = new_logits[-1, :]
                    else:
                        next_token_logits = new_logits
                    
                    next_token_id = torch.argmax(next_token_logits, dim=-1).item()
                    generated_tokens.append(next_token_id)
                    
                    token_text = tokenizer.decode([next_token_id], skip_special_tokens=True)
                    log_stage(f"Generated token {step+1}: {next_token_id} (text: '{token_text}')")
                    
                    # Send token back to first stage
                    communicator.send_token_id(next_token_id)
                    
                    # Check for EOS
                    if tokenizer.eos_token_id is not None and next_token_id == tokenizer.eos_token_id:
                        log_stage("Generated EOS token, stopping generation")
                        break
                
                # Decode the generated tokens
                output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                log_stage(f"Final generated text: '{output_text}'")
                
                if output_queue is not None:
                    output_queue.put(output_text)
            
            else:
                # Middle stages: receive, process, and forward
                log_stage("Waiting for initial activations...")
                
                # Receive initial activations
                tensor_dict = communicator.recv_tensor_dict()
                if tensor_dict is None:
                    raise RuntimeError("Failed to receive initial tensor dict")
                
                # Update pipeline state
                pipeline_state.update_from_tensor_dict(tensor_dict)
                initial_seq_len = pipeline_state.global_seq_len
                
                log_stage(f"Received initial activations. Global seq len: {initial_seq_len}")
                
                # Process initial activations
                hidden_states = tensor_dict["hidden_states"]
                residual = tensor_dict.get("residual")
                
                # Ensure correct dtype
                if hidden_states.dtype != model_weight_dtype:
                    hidden_states = hidden_states.to(dtype=model_weight_dtype)
                if residual is not None and residual.dtype != model_weight_dtype:
                    residual = residual.to(dtype=model_weight_dtype)
                
                # Get positions from tensor dict or create default
                positions = tensor_dict.get("token_positions")
                if positions is None:
                    positions = torch.arange(initial_seq_len, dtype=torch.long, device=hidden_states.device)
                
                # Process through model
                intermediate_tensors = IntermediateTensors({
                    "hidden_states": hidden_states,
                    "residual": residual
                })
                
                with set_current_vllm_config(vllm_config=vllm_config):
                    with set_forward_context(None, vllm_config, num_tokens=initial_seq_len):
                        output = model(
                            input_ids=None,
                            positions=positions,
                            intermediate_tensors=intermediate_tensors,
                        )
                
                # Forward to next stage
                if isinstance(output, IntermediateTensors):
                    hidden_states_out = output["hidden_states"]
                    residual_out = output.get("residual")
                    
                    # Prepare tensor dict with state information
                    tensor_dict_out = pipeline_state.prepare_tensor_dict(
                        hidden_states_out, residual_out, is_decode=False
                    )
                    
                    communicator.send_tensor_dict(tensor_dict_out)
                    log_stage("Forwarded initial activations to next stage")
                
                # Generation loop
                for step in range(max_new_tokens):
                    # Receive new hidden states
                    new_tensor_dict = communicator.recv_tensor_dict()
                    if new_tensor_dict is None:
                        log_stage("No more hidden states received, stopping processing")
                        break
                    
                    # Update pipeline state
                    pipeline_state.update_from_tensor_dict(new_tensor_dict)
                    
                    # Process new hidden states
                    hidden_states = new_tensor_dict["hidden_states"]
                    residual = new_tensor_dict.get("residual")
                    
                    # Ensure correct dtype
                    if hidden_states.dtype != model_weight_dtype:
                        hidden_states = hidden_states.to(dtype=model_weight_dtype)
                    if residual is not None and residual.dtype != model_weight_dtype:
                        residual = residual.to(dtype=model_weight_dtype)
                    
                    # Get position from tensor dict
                    current_position = new_tensor_dict.get("token_positions")
                    if current_position is None:
                        current_position = torch.tensor([pipeline_state.get_current_position()-1], device=hidden_states.device)
                    
                    # Process through model
                    new_intermediate_tensors = IntermediateTensors({
                        "hidden_states": hidden_states,
                        "residual": residual
                    })
                    
                    with set_current_vllm_config(vllm_config=vllm_config):
                        with set_forward_context(None, vllm_config, num_tokens=1):
                            new_output = model(
                                input_ids=None,
                                positions=current_position,
                                intermediate_tensors=new_intermediate_tensors,
                            )
                    
                    # Forward to next stage
                    if isinstance(new_output, IntermediateTensors):
                        hidden_states_out = new_output["hidden_states"]
                        residual_out = new_output.get("residual")
                        
                        # Prepare tensor dict with state information
                        tensor_dict_out = pipeline_state.prepare_tensor_dict(
                            hidden_states_out, residual_out, is_decode=True
                        )
                        
                        communicator.send_tensor_dict(tensor_dict_out)
                        log_stage(f"Forwarded hidden states for generation step {step+1}")
                
                log_stage("Middle stage processing complete")
        
        except Exception as e:
            log_stage(f"Error during processing: {str(e)}", "ERROR")
            import traceback
            traceback.print_exc()
            if output_queue is not None and is_last_stage:
                output_queue.put(f"ERROR: {str(e)}")
        
        finally:
            communicator.close()
            log_stage("Worker process exiting")
    
    except Exception as e:
        logger.error(f"Stage {stage_idx} fatal error: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        if output_queue is not None and stage_idx == num_stages - 1:
            try:
                output_queue.put(f"ERROR: Stage {stage_idx} failed: {e}")
            except:
                pass
    finally:
        logger.info(f"Stage {stage_idx} worker process exiting")


def setup_zeromq_pp_group(stage_idx: int, num_stages: int, communicator: ZeroMQCommunicator):
    """Setup ZeroMQ-based PP group and monkey patch get_pp_group"""
    global _zmq_communicator
    _zmq_communicator = communicator
    
    # Create mock PP group
    zmq_pp_group = ZeroMQPPGroup(stage_idx, num_stages, communicator)
    
    # Monkey patch get_pp_group to return our ZeroMQ group
    # Also set _PP so get_pp_group() doesn't assert
    import vllm.distributed.parallel_state as parallel_state_module
    original_get_pp_group = parallel_state_module.get_pp_group
    
    # Set _PP to our ZeroMQ group so get_pp_group() works
    parallel_state_module._PP = zmq_pp_group
    
    def get_zmq_pp_group():
        return zmq_pp_group
    
    parallel_state_module.get_pp_group = get_zmq_pp_group
    logger.info(f"Stage {stage_idx} patched get_pp_group to use ZeroMQ")


def stage_worker_process(
    stage_idx: int,
    pipeline_dir: str,
    num_stages: int,
    zmq_ports: list[int],
    device: str,
    dist_init_path: str,
    input_queue: Optional[multiprocessing.Queue],
    output_queue: Optional[multiprocessing.Queue],
    max_new_tokens: int = 256,
):
    """Stage worker process using vLLM engine with ZeroMQ communication"""
    import os
    
    try:
        # Set device for this stage
        if device.startswith("cuda") and torch.cuda.device_count() > 0:
            gpu_id = stage_idx % max(torch.cuda.device_count(), 1)
            stage_device = f"cuda:{gpu_id}"
            torch.cuda.set_device(torch.device(stage_device))
        else:
            stage_device = device
        
        log_file = Path(f"/tmp/pipeline_stage_{stage_idx}.log")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        # Clear previous log content
        try:
            log_file.write_text("", encoding="utf-8")
        except Exception:
            pass
        
        def log_stage(message: str) -> None:
            formatted = f"[Stage {stage_idx}] {message}"
            print(formatted, file=sys.stderr, flush=True)
            try:
                with open(log_file, "a", encoding="utf-8") as lf:
                    lf.write(formatted + "\n")
            except Exception:
                logger.debug("Failed to write stage log", exc_info=True)
            logger.info(formatted)
        
        log_stage(f"Worker process started on {stage_device} (PID: {os.getpid()})")
        
        # Avoid NCCL communicator requirements by disabling device communicators
        import vllm.distributed.parallel_state as ps
        
        if not hasattr(ps, "_orig_init_model_parallel_group"):
            ps._orig_init_model_parallel_group = ps.init_model_parallel_group
        
        def init_group_no_device(
            group_ranks,
            local_rank,
            backend,
            use_message_queue_broadcaster=False,
            group_name=None,
            use_device_communicator=False,
        ):
            return ps._orig_init_model_parallel_group(
                group_ranks=group_ranks,
                local_rank=local_rank,
                backend=backend,
                use_message_queue_broadcaster=use_message_queue_broadcaster,
                group_name=group_name,
                use_device_communicator=False,
            )
        
        ps.init_model_parallel_group = init_group_no_device
        
        # Initialize distributed environment (required by vLLM)
        # Use a shared file init method so all stages join the same rendezvous
        try:
            init_method = f"file://{dist_init_path}"
            backend = "gloo"
            log_stage(f"Initializing distributed environment via {init_method} (backend={backend})")
            init_distributed_environment(
                world_size=num_stages,  # Full pipeline size
                rank=stage_idx,  # Each stage has its own rank
                distributed_init_method=init_method,
                local_rank=0,
                backend=backend,
            )
            
            # Initialize model parallel
            # Each stage thinks it's part of a pipeline with num_stages stages
            # This makes the engine call send/recv_tensor_dict which we'll intercept
            initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=num_stages,  # Full pipeline size
            )
            
            logger.debug(f"Stage {stage_idx} initialized distributed environment")
            log_stage("Distributed environment initialized")
        except Exception as e:
            logger.warning(f"Stage {stage_idx} failed to initialize distributed environment: {e}", exc_info=True)
        
        # Setup ZeroMQ communicator BEFORE creating engine
        communicator = ZeroMQCommunicator(
            stage_idx=stage_idx,
            num_stages=num_stages,
            zmq_ports=zmq_ports,
            device=stage_device,
            timeout_ms=300000,  # 5 minutes timeout
        )
        log_stage("ZeroMQ communicator initialized")
        
        # Setup ZeroMQ PP group hook BEFORE creating engine
        # This must be done before the engine is created so it uses our hook
        setup_zeromq_pp_group(stage_idx, num_stages, communicator)
        log_stage("ZeroMQ PP group patched")
        
        # Load stage model directory
        log_stage(f"About to load model from pipeline_dir={pipeline_dir}")
        stage_dir = Path(pipeline_dir) / f"stage_{stage_idx}"
        if not stage_dir.exists():
            raise FileNotFoundError(f"Stage directory not found: {stage_dir}")
        
        log_stage(f"Loading vLLM model from {stage_dir}")
        logger.info(f"Stage {stage_idx} loading vLLM model from {stage_dir}")
        
        # Create vLLM model using model executor (similar to test_pipeline_vllm.py)
        # We can't use full LLM class because each stage is incomplete
        try:
            from vllm.config import ModelConfig, VllmConfig, LoadConfig, DeviceConfig, CompilationConfig, CompilationMode, ParallelConfig
            from vllm.model_executor.model_loader.utils import initialize_model
            from vllm.config import set_current_vllm_config
            from vllm.forward_context import set_forward_context, get_forward_context
            from transformers import AutoTokenizer
            
            # Load config
            import json
            config_path = stage_dir / "config.json"
            with open(config_path, "r", encoding="utf-8") as f:
                stage_config = json.load(f)
            
            pipeline_info = stage_config.get("_pipeline_info", {})
            stage_start_layer = pipeline_info.get("start_layer")
            stage_end_layer = pipeline_info.get("end_layer")
            
            # Load global_layer_map from pipeline_info (keys are strings in JSON, convert to int)
            global_layer_map = None
            if "global_layer_map" in pipeline_info:
                global_layer_map_serializable = pipeline_info["global_layer_map"]
                global_layer_map = {
                    int(local_idx): global_idx
                    for local_idx, global_idx in global_layer_map_serializable.items()
                }
                log_stage(f"Loaded global_layer_map with {len(global_layer_map)} entries")
            
            # Store global_layer_map on the model for use in patched_forward
            # This will be set after model is initialized
            
            # Get dtype
            if "dtype" in stage_config:
                dtype_str = stage_config["dtype"]
            elif "torch_dtype" in stage_config:
                dtype_str = stage_config["torch_dtype"]
            else:
                dtype_str = "float16"
            
            if dtype_str == "float16" or dtype_str == "torch.float16":
                model_dtype = "float16"
            elif dtype_str == "bfloat16" or dtype_str == "torch.bfloat16":
                model_dtype = "bfloat16"
            else:
                model_dtype = "float16"
            
            # Create ModelConfig
            model_config = ModelConfig(
                model=str(stage_dir),
                dtype=model_dtype,
                trust_remote_code=True,
                task=None,
                tokenizer=str(stage_dir),
                tokenizer_mode="auto",
                seed=0,
                quantization=None,
            )
            
            # Modify num_hidden_layers for this stage
            from vllm.distributed.utils import get_pp_indices
            if "_pipeline_info" in stage_config:
                original_num_layers = stage_config["_pipeline_info"].get("original_num_hidden_layers")
            else:
                original_num_layers = model_config.hf_config.num_hidden_layers
            
            start_layer, end_layer = get_pp_indices(
                original_num_layers,
                stage_idx,
                num_stages,
            )
            model_config.hf_config.num_hidden_layers = end_layer - start_layer
            if stage_start_layer is None:
                stage_start_layer = start_layer
            if stage_end_layer is None:
                stage_end_layer = end_layer
            
            # Create VllmConfig
            load_config = LoadConfig(load_format="pt", download_dir=None)
            # DeviceConfig accepts "cuda" or "cpu" string, or torch.device
            device_for_config = stage_device if stage_device != "cpu" else "cpu"
            device_config = DeviceConfig(device=device_for_config)
            compilation_config = CompilationConfig(mode=CompilationMode.NONE)
            parallel_config = ParallelConfig(
                pipeline_parallel_size=num_stages,
                tensor_parallel_size=1,
            )
            
            vllm_config = VllmConfig(
                model_config=model_config,
                load_config=load_config,
                device_config=device_config,
                compilation_config=compilation_config,
                parallel_config=parallel_config,
            )
            
            log_stage(f"Initializing model on device: {stage_device}")
            # Initialize model
            with set_current_vllm_config(vllm_config=vllm_config):
                model = initialize_model(
                    vllm_config=vllm_config,
                    model_config=model_config,
                )
            
            # Load weights directly to target device
            from vllm.model_executor.models.utils import AutoWeightsLoader
            weights_file = None
            if (stage_dir / "model.safetensors").exists():
                weights_file = str(stage_dir / "model.safetensors")
            elif (stage_dir / "pytorch_model.bin").exists():
                weights_file = str(stage_dir / "pytorch_model.bin")
            
            if weights_file:
                log_stage(f"Loading weights from {weights_file} to device {stage_device}")
                # Load weights directly to target device for faster loading
                target_device = stage_device if stage_device != "cpu" else "cpu"
                if weights_file.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    state_dict = load_file(weights_file, device=target_device)
                else:
                    state_dict = torch.load(weights_file, map_location=target_device, weights_only=True)
                
                # Convert all weights to fp16
                log_stage("Converting weights to fp16...")
                for key, value in state_dict.items():
                    if isinstance(value, torch.Tensor) and value.dtype != torch.float16:
                        state_dict[key] = value.to(dtype=torch.float16)
                
                weights_loader = AutoWeightsLoader(model)
                weights = [(k, v) for k, v in state_dict.items()]
                weights_loader.load_weights(weights)
                log_stage("Weights loaded in fp16")
            
            # Process weights after loading
            from vllm.model_executor.model_loader.utils import process_weights_after_loading
            target_torch_device = torch.device(stage_device if stage_device != "cpu" else "cpu")
            process_weights_after_loading(
                model,
                model_config,
                target_torch_device
            )
            
            # Ensure model is on correct device
            if stage_device != "cpu":
                model = model.to(stage_device)
                # Verify model parameters are on GPU
                first_param = next(model.parameters())
                log_stage(f"Model device check: first param on {first_param.device}")
                if first_param.device.type != "cuda":
                    log_stage(f"WARNING: Model parameters not on CUDA! Moving to {stage_device}")
                    model = model.to(stage_device)
            
            # Ensure all parameters are fp16
            first_param = next(model.parameters())
            if first_param.dtype != torch.float16:
                log_stage(f"Converting model parameters to fp16 (current: {first_param.dtype})...")
                for param in model.parameters():
                    if param.dtype != torch.float16:
                        param.data = param.data.to(dtype=torch.float16)
                log_stage("Model parameters converted to fp16")
            else:
                log_stage(f"Model parameters already in fp16")
            
            model.eval()
            stage_vllm_config = vllm_config
            
            # Store global_layer_map and stage_start_layer on the model for use in patched_forward
            if hasattr(model, "model"):
                model.model.global_layer_map = global_layer_map
                model.model.stage_start_layer = stage_start_layer or 0
                if global_layer_map:
                    log_stage(f"Stored global_layer_map on model with {len(global_layer_map)} entries")
                    
                    # Update attention layer's layer_name attributes to use global indices
                    # This ensures get_attention_context can find the correct entry in forward_context
                    base_model = model.model if hasattr(model, "model") else model
                    layers = getattr(base_model, "layers", None)
                    if layers is not None:
                        updated_count = 0
                        for local_idx, layer_module in enumerate(layers):
                            if local_idx in global_layer_map:
                                global_idx = global_layer_map[local_idx]
                                # Update attention layer's layer_name
                                if hasattr(layer_module, "self_attn") and hasattr(layer_module.self_attn, "attn"):
                                    attn_layer = layer_module.self_attn.attn
                                    if hasattr(attn_layer, "layer_name"):
                                        old_layer_name = attn_layer.layer_name
                                        new_layer_name = f"model.layers.{global_idx}.self_attn.attn"
                                        attn_layer.layer_name = new_layer_name
                                        updated_count += 1
                                        log_stage(f"Updated layer {local_idx} attention layer_name: {old_layer_name} -> {new_layer_name}")
                        log_stage(f"Updated {updated_count} attention layer layer_name attributes to use global indices")
                else:
                    log_stage("No global_layer_map available (will use local indices)")
            
            # Register attention layers for this stage
            stage_attention_layer_names = _register_stage_attention_layers(
                model=model,
                vllm_config=stage_vllm_config,
                stage_start_layer=stage_start_layer or 0,
                stage_end_layer=stage_end_layer,
                global_layer_map=global_layer_map,
                stage_idx=stage_idx,
            )
            if not stage_attention_layer_names:
                log_stage("WARNING: No attention layers registered; kv cache may be unavailable")
            
            # Ensure forward_context exists: set a default forward context if not present
            try:
                ctx = get_forward_context()
                log_stage("Forward context already set; skipping auto-init")
            except AssertionError:
                # Initialize a minimal forward context using set_forward_context
                try:
                    set_forward_context(None, stage_vllm_config, num_tokens=1)
                    log_stage("Initialized default forward context via set_forward_context(..., num_tokens=1)")
                except Exception as ex:
                    log_stage(f"Failed to initialize forward context: {ex}")
                    raise
            
            # Print forward_context keys snapshot for debugging
            try:
                ctx = get_forward_context()
                # Attempt to list keys
                keys_list = []
                if hasattr(ctx, 'no_compile_layers'):
                    keys_list = list(ctx.no_compile_layers.keys())
                elif hasattr(ctx, 'keys'):
                    keys_list = list(ctx.keys())
                log_stage(f"After init: forward_context keys ({len(keys_list)}): {keys_list[:40]}")
            except Exception as ex:
                log_stage(f"Could not introspect forward_context: {ex}")
            
            log_stage(f"Model initialized and on device: {next(model.parameters()).device}")
            
            # Load tokenizer (only first stage)
            tokenizer = None
            if stage_idx == 0:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(stage_dir), trust_remote_code=True
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
            
            # Monkey patch: 修复forward方法，让它根据intermediate_tensors判断，而不是get_pp_group()
            # 因为每个stage是独立进程，get_pp_group()无法正确工作
            if hasattr(model, "model") and hasattr(model.model, "forward"):
                original_forward = model.model.forward
                
                def patched_forward(
                    self_model,
                    input_ids=None,
                    positions=None,
                    intermediate_tensors=None,
                    inputs_embeds=None,
                    **kwargs
                ):
                    # 根据intermediate_tensors判断是否是第一阶段
                    is_first_rank = (intermediate_tensors is None and input_ids is not None)
                    
                    if is_first_rank:
                        if inputs_embeds is not None:
                            hidden_states = inputs_embeds
                        else:
                            hidden_states = self_model.embed_input_ids(input_ids)
                        residual = None
                    else:
                        assert intermediate_tensors is not None
                        hidden_states = intermediate_tensors["hidden_states"]
                        residual = intermediate_tensors["residual"]
                        # CRITICAL: Do NOT regenerate positions if they're provided!
                        # In pipeline, positions are GLOBAL and must be preserved.
                        # Only regenerate if positions is None or completely invalid.
                        if positions is None:
                            # Fallback: regenerate from local seq_length (should not happen in pipeline)
                            seq_length = hidden_states.shape[0] if hidden_states.dim() == 2 else hidden_states.shape[1]
                            positions = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device)
                            logger.warning(f"Stage {stage_idx}: positions was None, regenerated from local seq_length={seq_length} (may cause rotary emb errors)")
                        # If positions is provided, use it as-is (it's global position from previous stage)
                    
                    # 继续执行后续的layer处理
                    from itertools import islice
                    aux_hidden_states = []
                    # Get global_layer_map and stage_start_layer if available
                    global_layer_map = getattr(self_model, "global_layer_map", None)
                    stage_start_layer = getattr(self_model, "stage_start_layer", 0)
                    
                    # Iterate through layers using local indices
                    # Note: attention layer's layer_name has already been updated to global index during model loading
                    layers_slice = list(islice(self_model.layers, self_model.start_layer, self_model.end_layer))
                    for local_idx, layer in enumerate(layers_slice):
                        if local_idx in self_model.aux_hidden_state_layers:
                            aux_hidden_states.append(hidden_states + residual if residual is not None else hidden_states)
                        hidden_states, residual = layer(positions, hidden_states, residual)
                    
                    # 判断是否是最后stage
                    is_last_rank = (stage_idx == num_stages - 1)
                    
                    if not is_last_rank:
                        return IntermediateTensors({
                            "hidden_states": hidden_states,
                            "residual": residual
                        })
                    
                    # 最后stage：应用norm
                    expected_hidden_size = self_model.config.hidden_size
                    if hidden_states.shape[-1] != expected_hidden_size:
                        if hidden_states.numel() == hidden_states.shape[0] * expected_hidden_size:
                            hidden_states = hidden_states.reshape(hidden_states.shape[0], expected_hidden_size)
                    
                    if residual is not None and residual.shape[-1] != expected_hidden_size:
                        if residual.numel() == residual.shape[0] * expected_hidden_size:
                            residual = residual.reshape(residual.shape[0], expected_hidden_size)
                    
                    hidden_states, _ = self_model.norm(hidden_states, residual)
                    
                    if hidden_states.dim() > 2:
                        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
                    elif hidden_states.dim() == 1:
                        hidden_states = hidden_states.unsqueeze(0)
                    
                    if len(aux_hidden_states) > 0:
                        return hidden_states, aux_hidden_states
                    return hidden_states
                
                # 绑定patched_forward到模型实例
                import types
                model.model.forward = types.MethodType(patched_forward, model.model)
                logger.debug(f"Stage {stage_idx} patched forward method to handle pipeline stages correctly")
            
            log_stage("vLLM model loaded successfully")
            logger.info(f"Stage {stage_idx} vLLM model loaded successfully")
            
        except Exception as e:
            logger.error(f"Stage {stage_idx} failed to load vLLM model: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
            raise
        
        is_first = stage_idx == 0
        is_last = stage_idx == num_stages - 1
        
        if is_first:
            # First stage: receive input, process, send to next stage
            log_stage("Waiting for input from queue...")
            logger.info(f"Stage {stage_idx} waiting for input...")
            
            try:
                input_data = input_queue.get(timeout=300)
                if input_data is None:
                    logger.info(f"Stage {stage_idx} received exit signal")
                    communicator.close()
                    return
                
                input_text = input_data
                logger.info(f"Stage {stage_idx} received input: {input_text}")
                
                # Tokenize input
                inputs = tokenizer(input_text, return_tensors="pt", padding=True)
                inputs = {k: v.to(stage_device) for k, v in inputs.items()}
                batch_size, seq_length = inputs["input_ids"].shape
                
                # Use model to process input
                # The ZeroMQ hook will intercept IntermediateTensors and send them
                # via ZeroMQ to the next stage
                log_stage("Processing input through model...")
                
                # Process initial prompt
                # Flatten input_ids for vLLM format
                input_ids = inputs["input_ids"].flatten()
                # Global positions: [0, 1, 2, ..., seq_len-1] for initial prompt
                global_positions = torch.arange(batch_size * seq_length, dtype=torch.long, device=stage_device)
                
                # Call model forward - ZeroMQ hook will intercept IntermediateTensors
                
                with set_current_vllm_config(vllm_config=stage_vllm_config):
                    with set_forward_context(
                        None, stage_vllm_config, num_tokens=batch_size * seq_length
                    ):
                        output = model(
                            input_ids=input_ids,
                            positions=global_positions,
                            intermediate_tensors=None,
                        )
                
                # Process output: if IntermediateTensors, send via ZeroMQ
                if isinstance(output, IntermediateTensors):
                    # Ensure dtype consistency and pass global positions
                    hidden_states = output["hidden_states"]
                    residual = output["residual"]
                    
                    # CRITICAL: Get actual dtype from model weights for sending
                    # We should send in the dtype that matches the next stage's model weights
                    # But for now, use current model's weight dtype to ensure consistency
                    model_weight_dtype = None
                    if hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > 0:
                        first_layer = model.model.layers[0]
                        if hasattr(first_layer, "self_attn") and hasattr(first_layer.self_attn, "q_proj"):
                            model_weight_dtype = first_layer.self_attn.q_proj.weight.dtype
                        elif hasattr(first_layer, "mlp") and hasattr(first_layer.mlp, "gate_proj"):
                            model_weight_dtype = first_layer.mlp.gate_proj.weight.dtype
                    
                    if model_weight_dtype is None:
                        first_param = next(model.parameters())
                        model_weight_dtype = first_param.dtype
                    
                    # PRINT: 上一层输出时的精度
                    log_stage(f"[OUTPUT] Stage {stage_idx} - Model weight dtype: {model_weight_dtype}")
                    log_stage(f"[OUTPUT] Stage {stage_idx} - hidden_states dtype BEFORE conversion: {hidden_states.dtype}, shape: {hidden_states.shape}")
                    if residual is not None:
                        log_stage(f"[OUTPUT] Stage {stage_idx} - residual dtype BEFORE conversion: {residual.dtype}, shape: {residual.shape}")
                    
                    # Convert to match model weight dtype before sending
                    if hidden_states.dtype != model_weight_dtype:
                        log_stage(f"Converting hidden_states dtype from {hidden_states.dtype} to {model_weight_dtype} before sending")
                        hidden_states = hidden_states.to(dtype=model_weight_dtype)
                    if residual is not None and residual.dtype != model_weight_dtype:
                        log_stage(f"Converting residual dtype from {residual.dtype} to {model_weight_dtype} before sending")
                        residual = residual.to(dtype=model_weight_dtype)
                    
                    # PRINT: 上一层输出时的精度（转换后）
                    log_stage(f"[OUTPUT] Stage {stage_idx} - hidden_states dtype AFTER conversion: {hidden_states.dtype}, shape: {hidden_states.shape}")
                    if residual is not None:
                        log_stage(f"[OUTPUT] Stage {stage_idx} - residual dtype AFTER conversion: {residual.dtype}, shape: {residual.shape}")
                    
                    tensor_dict = {
                        "hidden_states": hidden_states,
                        "residual": residual,
                        "global_positions": global_positions,  # Pass global positions to next stage
                        "is_decode": False,  # Mark as initial prompt, not decode
                    }
                    if not is_last:
                        communicator.send_tensor_dict(tensor_dict)
                        logger.info(f"Stage {stage_idx} sent initial hidden states to next stage with global positions")
                    else:
                        # Single stage: process directly
                        if hasattr(model, "compute_logits"):
                            logits = model.compute_logits(output["hidden_states"])
                        else:
                            logits = output["hidden_states"]
                        
                        # Extract first token
                        if hasattr(logits, "shape"):
                            if len(logits.shape) == 3:
                                next_token_logits = logits[0, -1, :]
                            elif len(logits.shape) == 2:
                                if logits.shape[0] > logits.shape[1]:
                                    next_token_logits = logits[-1, :]
                                else:
                                    next_token_logits = logits[0, :] if logits.shape[0] == 1 else logits[-1, :]
                            else:
                                next_token_logits = logits.flatten()
                        else:
                            next_token_logits = logits
                        
                        next_token_id = torch.argmax(next_token_logits, dim=-1)
                        if isinstance(next_token_id, torch.Tensor):
                            next_token_id = next_token_id.item()
                        
                        generated_text = tokenizer.decode([next_token_id], skip_special_tokens=True)
                        if output_queue is not None:
                            output_queue.put(generated_text)
                        communicator.close()
                        return
                elif isinstance(output, torch.Tensor) and is_last:
                    # Single stage: output is logits
                    if hasattr(model, "compute_logits"):
                        logits = model.compute_logits(output)
                    else:
                        logits = output
                    
                    # Extract first token
                    if hasattr(logits, "shape"):
                        if len(logits.shape) == 3:
                            next_token_logits = logits[0, -1, :]
                        elif len(logits.shape) == 2:
                            if logits.shape[0] > logits.shape[1]:
                                next_token_logits = logits[-1, :]
                            else:
                                next_token_logits = logits[0, :] if logits.shape[0] == 1 else logits[-1, :]
                        else:
                            next_token_logits = logits.flatten()
                    else:
                        next_token_logits = logits
                    
                    next_token_id = torch.argmax(next_token_logits, dim=-1)
                    if isinstance(next_token_id, torch.Tensor):
                        next_token_id = next_token_id.item()
                    
                    generated_text = tokenizer.decode([next_token_id], skip_special_tokens=True)
                    if output_queue is not None:
                        output_queue.put(generated_text)
                    communicator.close()
                    return
                
                # Handle autoregressive generation (multi-stage)
                generation_step = 0
                while True:
                    token_id = communicator.recv_token_id()
                    if token_id is None:
                        logger.info(f"Stage {stage_idx} no more tokens to process (generation complete)")
                        break
                    
                    generation_step += 1
                    logger.info(f"Stage {stage_idx} received token ID {token_id} for generation step {generation_step}")
                    
                    # Process new token with KV cache
                    new_token_tensor = torch.tensor([[token_id]], dtype=torch.long, device=stage_device)
                    new_input_ids = new_token_tensor.flatten()
                    current_seq_len = seq_length + generation_step
                    # Global position for decode: seq_length + generation_step - 1
                    global_decode_position = seq_length + generation_step - 1
                    new_positions = torch.tensor([global_decode_position], dtype=torch.long, device=stage_device)
                    
                    # Call model forward with KV cache
                    # NOTE: In pipeline mode, we don't set complex attention metadata
                    # because each stage is independent and KV cache management is complex.
                    # The model will handle attention without explicit metadata.
                    # Setting None allows the model to work in a simpler mode.
                    
                    with set_current_vllm_config(vllm_config=stage_vllm_config):
                        with set_forward_context(
                            None, stage_vllm_config, num_tokens=1
                        ):
                            new_output = model(
                                input_ids=new_input_ids,
                                positions=new_positions,
                                intermediate_tensors=None,
                            )
                    
                    # Process output
                    if isinstance(new_output, IntermediateTensors):
                        hidden_states = new_output["hidden_states"]
                        residual = new_output["residual"]
                        
                        # CRITICAL: Get actual dtype from model weights for sending
                        model_weight_dtype = None
                        if hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > 0:
                            first_layer = model.model.layers[0]
                            if hasattr(first_layer, "self_attn") and hasattr(first_layer.self_attn, "q_proj"):
                                model_weight_dtype = first_layer.self_attn.q_proj.weight.dtype
                            elif hasattr(first_layer, "mlp") and hasattr(first_layer.mlp, "gate_proj"):
                                model_weight_dtype = first_layer.mlp.gate_proj.weight.dtype
                        
                        if model_weight_dtype is None:
                            first_param = next(model.parameters())
                            model_weight_dtype = first_param.dtype
                        
                        # PRINT: 上一层输出时的精度（decode）
                        log_stage(f"[OUTPUT-DECODE] Stage {stage_idx} step {generation_step} - Model weight dtype: {model_weight_dtype}")
                        log_stage(f"[OUTPUT-DECODE] Stage {stage_idx} step {generation_step} - hidden_states dtype BEFORE conversion: {hidden_states.dtype}, shape: {hidden_states.shape}")
                        if residual is not None:
                            log_stage(f"[OUTPUT-DECODE] Stage {stage_idx} step {generation_step} - residual dtype BEFORE conversion: {residual.dtype}, shape: {residual.shape}")
                        
                        # Convert to match model weight dtype before sending
                        if hidden_states.dtype != model_weight_dtype:
                            log_stage(f"Converting hidden_states dtype from {hidden_states.dtype} to {model_weight_dtype} before sending")
                            hidden_states = hidden_states.to(dtype=model_weight_dtype)
                        if residual is not None and residual.dtype != model_weight_dtype:
                            log_stage(f"Converting residual dtype from {residual.dtype} to {model_weight_dtype} before sending")
                            residual = residual.to(dtype=model_weight_dtype)
                        
                        # PRINT: 上一层输出时的精度（转换后，decode）
                        log_stage(f"[OUTPUT-DECODE] Stage {stage_idx} step {generation_step} - hidden_states dtype AFTER conversion: {hidden_states.dtype}, shape: {hidden_states.shape}")
                        if residual is not None:
                            log_stage(f"[OUTPUT-DECODE] Stage {stage_idx} step {generation_step} - residual dtype AFTER conversion: {residual.dtype}, shape: {residual.shape}")
                        
                        new_tensor_dict = {
                            "hidden_states": hidden_states,
                            "residual": residual,
                            "global_positions": new_positions,  # Pass global decode position
                            "is_decode": True,  # Mark as decode step
                        }
                        if not is_last:
                            communicator.send_tensor_dict(new_tensor_dict)
                            logger.info(f"Stage {stage_idx} sent new hidden state for token {token_id} at global position {global_decode_position}")
                
            except Exception as e:
                logger.error(f"Stage {stage_idx} error: {e}", exc_info=True)
                import traceback
                traceback.print_exc()
                if output_queue is not None:
                    output_queue.put(f"ERROR: {e}")
            
            communicator.close()
            
        elif is_last:
            # Last stage: receive from previous stage, generate output
            log_stage("Waiting to receive activations...")
            logger.info(f"Stage {stage_idx} waiting to receive activations")
            
            try:
                # Load tokenizer for decoding (from stage 0)
                stage0_dir = Path(pipeline_dir) / "stage_0"
                tokenizer = AutoTokenizer.from_pretrained(str(stage0_dir), trust_remote_code=True)
                
                # Use loaded model (already in scope)
                
                # Receive initial activations
                tensor_dict = communicator.recv_tensor_dict()
                if tensor_dict is None:
                    raise RuntimeError("Failed to receive tensor dict")
                
                hidden_states = tensor_dict["hidden_states"]
                residual = tensor_dict.get("residual")
                # Get global positions from previous stage (CRITICAL for rotary embeddings)
                global_positions = tensor_dict.get("global_positions")
                is_decode = tensor_dict.get("is_decode", False)
                
                logger.info(f"Stage {stage_idx} received activations")
                log_stage(f"Received activations, shape: {hidden_states.shape}, is_decode={is_decode}")
                
                # PRINT: 下一层接收时的精度（接收后，转换前）
                log_stage(f"[INPUT] Stage {stage_idx} - hidden_states dtype RECEIVED: {hidden_states.dtype}, shape: {hidden_states.shape}")
                if residual is not None:
                    log_stage(f"[INPUT] Stage {stage_idx} - residual dtype RECEIVED: {residual.dtype}, shape: {residual.shape}")
                
                # CRITICAL: Get actual dtype from model weights, not from config
                # Model weights may have different dtype than config (e.g., loaded as float32)
                # We must match the actual weight dtype to avoid "weight.scalar_type() == input.scalar_type()" errors
                # Try to get dtype from the first layer that will process hidden_states (more reliable)
                model_weight_dtype = None
                if hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > 0:
                    # Get dtype from first transformer layer (most reliable)
                    first_layer = model.model.layers[0]
                    if hasattr(first_layer, "self_attn") and hasattr(first_layer.self_attn, "q_proj"):
                        model_weight_dtype = first_layer.self_attn.q_proj.weight.dtype
                    elif hasattr(first_layer, "mlp") and hasattr(first_layer.mlp, "gate_proj"):
                        model_weight_dtype = first_layer.mlp.gate_proj.weight.dtype
                
                # Fallback to first parameter if layer check failed
                if model_weight_dtype is None:
                    first_param = next(model.parameters())
                    model_weight_dtype = first_param.dtype
                
                # PRINT: 推理精度（模型权重 dtype）
                log_stage(f"[INPUT] Stage {stage_idx} - Model weight dtype: {model_weight_dtype}")
                log_stage(f"[INPUT] Stage {stage_idx} - hidden_states dtype BEFORE conversion: {hidden_states.dtype}")
                
                # Convert hidden_states to match model weight dtype
                if hidden_states.dtype != model_weight_dtype:
                    log_stage(f"Converting hidden_states dtype from {hidden_states.dtype} to {model_weight_dtype} to match model weights")
                    hidden_states = hidden_states.to(dtype=model_weight_dtype)
                if residual is not None and residual.dtype != model_weight_dtype:
                    log_stage(f"Converting residual dtype from {residual.dtype} to {model_weight_dtype} to match model weights")
                    residual = residual.to(dtype=model_weight_dtype)
                
                # PRINT: 下一层接收时的精度（转换后）
                log_stage(f"[INPUT] Stage {stage_idx} - hidden_states dtype AFTER conversion: {hidden_states.dtype}, shape: {hidden_states.shape}")
                if residual is not None:
                    log_stage(f"[INPUT] Stage {stage_idx} - residual dtype AFTER conversion: {residual.dtype}, shape: {residual.shape}")
                
                # Get initial sequence length
                if hidden_states.dim() == 2:
                    initial_seq_len = hidden_states.shape[0]
                elif hidden_states.dim() == 3:
                    initial_seq_len = hidden_states.shape[1]
                else:
                    initial_seq_len = hidden_states.shape[0] if hidden_states.dim() > 0 else 1
                
                # Use global positions from previous stage (CRITICAL!)
                if global_positions is not None:
                    positions = global_positions.to(hidden_states.device)
                    log_stage(f"Using global positions from previous stage: {positions.tolist()}")
                else:
                    # Fallback: regenerate (should not happen, but handle gracefully)
                    log_stage("WARNING: No global positions received, regenerating from local seq_len (may cause rotary emb errors)")
                    positions = torch.arange(initial_seq_len, dtype=torch.long, device=hidden_states.device)
                
                # Process initial hidden states through model
                
                intermediate_tensors = IntermediateTensors({
                    "hidden_states": hidden_states,
                    "residual": residual
                })
                
                with set_current_vllm_config(vllm_config=stage_vllm_config):
                    with set_forward_context(
                        None, stage_vllm_config, num_tokens=initial_seq_len
                    ):
                        output = model(
                            input_ids=None,
                            positions=positions,
                            intermediate_tensors=intermediate_tensors,
                        )
                
                # Compute logits
                if isinstance(output, torch.Tensor):
                    if hasattr(model, "compute_logits"):
                        logits = model.compute_logits(output)
                    else:
                        logits = output
                else:
                    logits = output
                
                # Extract first token
                if hasattr(logits, "shape"):
                    if len(logits.shape) == 3:
                        next_token_logits = logits[0, -1, :]
                    elif len(logits.shape) == 2:
                        if logits.shape[0] > logits.shape[1]:
                            next_token_logits = logits[-1, :]
                        else:
                            next_token_logits = logits[0, :] if logits.shape[0] == 1 else logits[-1, :]
                    else:
                        next_token_logits = logits.flatten()
                else:
                    next_token_logits = logits
                
                next_token_id = torch.argmax(next_token_logits, dim=-1)
                if isinstance(next_token_id, torch.Tensor):
                    next_token_id = next_token_id.item()
                
                generated_tokens = [next_token_id]
                log_stage(f"Generated first token: {next_token_id} (text: '{tokenizer.decode([next_token_id])}')")
                
                # Autoregressive generation loop
                if tokenizer.eos_token_id is None or next_token_id != tokenizer.eos_token_id:
                    # Send first token ID back to stage 0 to start the generation loop
                    communicator.send_token_id(next_token_id)
                    log_stage(f"Sent first token ID {next_token_id} back to stage 0")
                    
                    for step in range(1, max_new_tokens):
                        # Send token ID back to stage 0 (for subsequent tokens)
                        communicator.send_token_id(next_token_id)
                        
                        # Receive new hidden states
                        new_tensor_dict = communicator.recv_tensor_dict()
                        if new_tensor_dict is None:
                            logger.error(f"Stage {stage_idx} failed to receive new hidden states at step {step}")
                            break
                        
                        new_hidden_states = new_tensor_dict["hidden_states"]
                        new_residual = new_tensor_dict.get("residual")
                        # Get global decode position from previous stage (CRITICAL!)
                        global_decode_positions = new_tensor_dict.get("global_positions")
                        is_decode_step = new_tensor_dict.get("is_decode", True)
                        
                        # PRINT: 下一层接收时的精度（decode，接收后，转换前）
                        log_stage(f"[INPUT-DECODE] Stage {stage_idx} step {step} - hidden_states dtype RECEIVED: {new_hidden_states.dtype}, shape: {new_hidden_states.shape}")
                        if new_residual is not None:
                            log_stage(f"[INPUT-DECODE] Stage {stage_idx} step {step} - residual dtype RECEIVED: {new_residual.dtype}, shape: {new_residual.shape}")
                        
                        # CRITICAL: Get actual dtype from model weights, not from config
                        model_weight_dtype = None
                        if hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > 0:
                            first_layer = model.model.layers[0]
                            if hasattr(first_layer, "self_attn") and hasattr(first_layer.self_attn, "q_proj"):
                                model_weight_dtype = first_layer.self_attn.q_proj.weight.dtype
                            elif hasattr(first_layer, "mlp") and hasattr(first_layer.mlp, "gate_proj"):
                                model_weight_dtype = first_layer.mlp.gate_proj.weight.dtype
                        
                        if model_weight_dtype is None:
                            first_param = next(model.parameters())
                            model_weight_dtype = first_param.dtype
                        
                        # PRINT: 推理精度（模型权重 dtype，decode）
                        log_stage(f"[INPUT-DECODE] Stage {stage_idx} step {step} - Model weight dtype: {model_weight_dtype}")
                        log_stage(f"[INPUT-DECODE] Stage {stage_idx} step {step} - hidden_states dtype BEFORE conversion: {new_hidden_states.dtype}")
                        
                        # Convert to match model weight dtype
                        if new_hidden_states.dtype != model_weight_dtype:
                            log_stage(f"Converting new_hidden_states dtype from {new_hidden_states.dtype} to {model_weight_dtype} to match model weights")
                            new_hidden_states = new_hidden_states.to(dtype=model_weight_dtype)
                        if new_residual is not None and new_residual.dtype != model_weight_dtype:
                            log_stage(f"Converting new_residual dtype from {new_residual.dtype} to {model_weight_dtype} to match model weights")
                            new_residual = new_residual.to(dtype=model_weight_dtype)
                        
                        # PRINT: 下一层接收时的精度（转换后，decode）
                        log_stage(f"[INPUT-DECODE] Stage {stage_idx} step {step} - hidden_states dtype AFTER conversion: {new_hidden_states.dtype}, shape: {new_hidden_states.shape}")
                        if new_residual is not None:
                            log_stage(f"[INPUT-DECODE] Stage {stage_idx} step {step} - residual dtype AFTER conversion: {new_residual.dtype}, shape: {new_residual.shape}")
                        
                        # Use global decode position from previous stage
                        if global_decode_positions is not None:
                            decode_position = global_decode_positions.to(new_hidden_states.device)
                            log_stage(f"Using global decode position from previous stage: {decode_position.tolist()}")
                        else:
                            # Fallback: calculate from local (should not happen)
                            log_stage("WARNING: No global decode position received, calculating from local (may cause rotary emb errors)")
                            current_seq_len = initial_seq_len + step
                            decode_position = torch.tensor([initial_seq_len + step - 1], dtype=torch.long, device=new_hidden_states.device)
                        
                        # Process through model
                        current_seq_len = initial_seq_len + step
                        
                        new_intermediate_tensors = IntermediateTensors({
                            "hidden_states": new_hidden_states,
                            "residual": new_residual
                        })
                        
                        # NOTE: In pipeline mode, we don't set complex attention metadata
                        # because each stage is independent and KV cache management is complex.
                        # The model will handle attention without explicit metadata.
                        
                        with set_current_vllm_config(vllm_config=stage_vllm_config):
                            with set_forward_context(
                                None, stage_vllm_config, num_tokens=1
                            ):
                                new_output = model(
                                    input_ids=None,
                                    positions=decode_position,
                                    intermediate_tensors=new_intermediate_tensors,
                                )
                        
                        # Compute logits
                        if isinstance(new_output, torch.Tensor):
                            if hasattr(model, "compute_logits"):
                                new_logits = model.compute_logits(new_output)
                            else:
                                new_logits = new_output
                        else:
                            new_logits = new_output
                        
                        # Extract next token
                        if hasattr(new_logits, "shape"):
                            if len(new_logits.shape) == 3:
                                next_token_logits = new_logits[0, -1, :]
                            elif len(new_logits.shape) == 2:
                                if new_logits.shape[0] > new_logits.shape[1]:
                                    next_token_logits = new_logits[-1, :]
                                else:
                                    next_token_logits = new_logits[0, :] if new_logits.shape[0] == 1 else new_logits[-1, :]
                            else:
                                next_token_logits = new_logits.flatten()
                        else:
                            next_token_logits = new_logits
                        
                        next_token_id = torch.argmax(next_token_logits, dim=-1)
                        if isinstance(next_token_id, torch.Tensor):
                            next_token_id = next_token_id.item()
                        
                        generated_tokens.append(next_token_id)
                        log_stage(f"Generated token {step+1}: {next_token_id} (text: '{tokenizer.decode([next_token_id])}')")
                        
                        # Check for EOS
                        if tokenizer.eos_token_id is not None and next_token_id == tokenizer.eos_token_id:
                            log_stage("Generated EOS token, stopping")
                            break
                
                # Decode all tokens
                output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                log_stage(f"Generated {len(generated_tokens)} tokens: '{output_text[:100]}...'")
                
                if output_queue is not None:
                    output_queue.put(output_text)
                
            except Exception as e:
                logger.error(f"Stage {stage_idx} error: {e}", exc_info=True)
                import traceback
                traceback.print_exc()
                if output_queue is not None:
                    output_queue.put(f"ERROR: {e}")
            
            communicator.close()
            
        else:
            # Middle stages: receive from previous, process, send to next
            log_stage("Waiting to receive activations...")
            logger.info(f"Stage {stage_idx} waiting to receive activations")
            
            try:
                # Use loaded model (already in scope)
                
                # Receive initial activations
                log_stage(f"Waiting to receive initial activations from previous stage...")
                tensor_dict = communicator.recv_tensor_dict()
                if tensor_dict is None:
                    raise RuntimeError("Failed to receive tensor dict")
                
                hidden_states = tensor_dict["hidden_states"]
                residual = tensor_dict.get("residual")
                # Get global positions from previous stage (CRITICAL for rotary embeddings)
                global_positions = tensor_dict.get("global_positions")
                is_decode = tensor_dict.get("is_decode", False)
                
                logger.info(f"Stage {stage_idx} received activations, processing...")
                log_stage(f"Received activations, shape: {hidden_states.shape}, residual={'present' if residual is not None else 'None'}, is_decode={is_decode}")
                
                # PRINT: 下一层接收时的精度（接收后，转换前）
                log_stage(f"[INPUT] Stage {stage_idx} - hidden_states dtype RECEIVED: {hidden_states.dtype}, shape: {hidden_states.shape}")
                if residual is not None:
                    log_stage(f"[INPUT] Stage {stage_idx} - residual dtype RECEIVED: {residual.dtype}, shape: {residual.shape}")
                
                # CRITICAL: Get actual dtype from model weights, not from config
                # Model weights may have different dtype than config (e.g., loaded as float32)
                # We must match the actual weight dtype to avoid "weight.scalar_type() == input.scalar_type()" errors
                # Try to get dtype from the first layer that will process hidden_states (more reliable)
                model_weight_dtype = None
                if hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > 0:
                    # Get dtype from first transformer layer (most reliable)
                    first_layer = model.model.layers[0]
                    if hasattr(first_layer, "self_attn") and hasattr(first_layer.self_attn, "q_proj"):
                        model_weight_dtype = first_layer.self_attn.q_proj.weight.dtype
                    elif hasattr(first_layer, "mlp") and hasattr(first_layer.mlp, "gate_proj"):
                        model_weight_dtype = first_layer.mlp.gate_proj.weight.dtype
                
                # Fallback to first parameter if layer check failed
                if model_weight_dtype is None:
                    first_param = next(model.parameters())
                    model_weight_dtype = first_param.dtype
                
                # PRINT: 推理精度（模型权重 dtype）
                log_stage(f"[INPUT] Stage {stage_idx} - Model weight dtype: {model_weight_dtype}")
                log_stage(f"[INPUT] Stage {stage_idx} - hidden_states dtype BEFORE conversion: {hidden_states.dtype}")
                
                # Convert hidden_states to match model weight dtype
                if hidden_states.dtype != model_weight_dtype:
                    log_stage(f"Converting hidden_states dtype from {hidden_states.dtype} to {model_weight_dtype} to match model weights")
                    hidden_states = hidden_states.to(dtype=model_weight_dtype)
                if residual is not None and residual.dtype != model_weight_dtype:
                    log_stage(f"Converting residual dtype from {residual.dtype} to {model_weight_dtype} to match model weights")
                    residual = residual.to(dtype=model_weight_dtype)
                
                # PRINT: 下一层接收时的精度（转换后）
                log_stage(f"[INPUT] Stage {stage_idx} - hidden_states dtype AFTER conversion: {hidden_states.dtype}, shape: {hidden_states.shape}")
                if residual is not None:
                    log_stage(f"[INPUT] Stage {stage_idx} - residual dtype AFTER conversion: {residual.dtype}, shape: {residual.shape}")
                
                # Get initial sequence length
                if hidden_states.dim() == 2:
                    initial_seq_len = hidden_states.shape[0]
                elif hidden_states.dim() == 3:
                    initial_seq_len = hidden_states.shape[1]
                else:
                    initial_seq_len = hidden_states.shape[0] if hidden_states.dim() > 0 else 1
                
                # Use global positions from previous stage (CRITICAL!)
                if global_positions is not None:
                    positions = global_positions.to(hidden_states.device)
                    log_stage(f"Using global positions from previous stage: {positions.tolist()}")
                else:
                    # Fallback: regenerate (should not happen, but handle gracefully)
                    log_stage("WARNING: No global positions received, regenerating from local seq_len (may cause rotary emb errors)")
                    positions = torch.arange(initial_seq_len, dtype=torch.long, device=hidden_states.device)
                
                # Process through model
                
                intermediate_tensors = IntermediateTensors({
                    "hidden_states": hidden_states,
                    "residual": residual
                })
                
                with set_current_vllm_config(vllm_config=stage_vllm_config):
                    with set_forward_context(
                        None, stage_vllm_config, num_tokens=initial_seq_len
                    ):
                        output = model(
                            input_ids=None,
                            positions=positions,
                            intermediate_tensors=intermediate_tensors,
                        )
                
                # Send output to next stage
                # Always manually send to ensure it's sent (ZeroMQ hook may not be active)
                if isinstance(output, IntermediateTensors):
                    # IntermediateTensors supports dict-like access with []
                    hidden_states = output["hidden_states"]
                    residual = output["residual"] if "residual" in output.tensors else None
                    
                    # CRITICAL: Get actual dtype from model weights for sending
                    # We should send in the dtype that matches the next stage's model weights
                    # But for now, use current model's weight dtype to ensure consistency
                    model_weight_dtype = None
                    if hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > 0:
                        first_layer = model.model.layers[0]
                        if hasattr(first_layer, "self_attn") and hasattr(first_layer.self_attn, "q_proj"):
                            model_weight_dtype = first_layer.self_attn.q_proj.weight.dtype
                        elif hasattr(first_layer, "mlp") and hasattr(first_layer.mlp, "gate_proj"):
                            model_weight_dtype = first_layer.mlp.gate_proj.weight.dtype
                    
                    if model_weight_dtype is None:
                        first_param = next(model.parameters())
                        model_weight_dtype = first_param.dtype
                    
                    # PRINT: 上一层输出时的精度（中间 stage，初始）
                    log_stage(f"[OUTPUT] Stage {stage_idx} - Model weight dtype: {model_weight_dtype}")
                    log_stage(f"[OUTPUT] Stage {stage_idx} - hidden_states dtype BEFORE conversion: {hidden_states.dtype}, shape: {hidden_states.shape}")
                    if residual is not None:
                        log_stage(f"[OUTPUT] Stage {stage_idx} - residual dtype BEFORE conversion: {residual.dtype}, shape: {residual.shape}")
                    
                    # Convert to match model weight dtype before sending
                    if hidden_states.dtype != model_weight_dtype:
                        log_stage(f"Converting hidden_states dtype from {hidden_states.dtype} to {model_weight_dtype} before sending")
                        hidden_states = hidden_states.to(dtype=model_weight_dtype)
                    if residual is not None and residual.dtype != model_weight_dtype:
                        log_stage(f"Converting residual dtype from {residual.dtype} to {model_weight_dtype} before sending")
                        residual = residual.to(dtype=model_weight_dtype)
                    
                    # PRINT: 上一层输出时的精度（转换后，中间 stage，初始）
                    log_stage(f"[OUTPUT] Stage {stage_idx} - hidden_states dtype AFTER conversion: {hidden_states.dtype}, shape: {hidden_states.shape}")
                    if residual is not None:
                        log_stage(f"[OUTPUT] Stage {stage_idx} - residual dtype AFTER conversion: {residual.dtype}, shape: {residual.shape}")
                    
                    tensor_dict = {
                        "hidden_states": hidden_states,
                        "residual": residual,
                        "global_positions": positions,  # Pass global positions to next stage
                        "is_decode": is_decode,  # Pass decode flag
                    }
                    log_stage(f"Sending IntermediateTensors to next stage: hidden_states shape={tensor_dict['hidden_states'].shape}")
                    communicator.send_tensor_dict(tensor_dict)
                else:
                    # Convert to tensor dict and send
                    hidden_states = output
                    
                    # CRITICAL: Get actual dtype from model weights for sending
                    first_param = next(model.parameters())
                    model_weight_dtype = first_param.dtype
                    
                    # Convert to match model weight dtype before sending
                    if hidden_states.dtype != model_weight_dtype:
                        log_stage(f"Converting hidden_states dtype from {hidden_states.dtype} to {model_weight_dtype} before sending")
                        hidden_states = hidden_states.to(dtype=model_weight_dtype)
                    
                    tensor_dict = {
                        "hidden_states": hidden_states,
                        "global_positions": positions,
                        "is_decode": is_decode,
                    }
                    log_stage(f"Sending tensor output to next stage: shape={output.shape}")
                    communicator.send_tensor_dict(tensor_dict)
                
                logger.info(f"Stage {stage_idx} sent initial hidden states to next stage")
                log_stage(f"Successfully sent initial hidden states to next stage")
                
                # Handle autoregressive generation
                generation_step = 0
                while True:
                    tensor_dict = communicator.recv_tensor_dict()
                    if tensor_dict is None:
                        logger.info(f"Stage {stage_idx} no more hidden states to process")
                        break
                    
                    generation_step += 1
                    hidden_states = tensor_dict["hidden_states"]
                    residual = tensor_dict.get("residual")
                    # Get global decode position from previous stage (CRITICAL!)
                    global_decode_positions = tensor_dict.get("global_positions")
                    is_decode_step = tensor_dict.get("is_decode", True)
                    
                    # CRITICAL: Get actual dtype from model weights and convert if needed
                    model_weight_dtype = None
                    if hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > 0:
                        first_layer = model.model.layers[0]
                        if hasattr(first_layer, "self_attn") and hasattr(first_layer.self_attn, "q_proj"):
                            model_weight_dtype = first_layer.self_attn.q_proj.weight.dtype
                        elif hasattr(first_layer, "mlp") and hasattr(first_layer.mlp, "gate_proj"):
                            model_weight_dtype = first_layer.mlp.gate_proj.weight.dtype
                    
                    if model_weight_dtype is None:
                        first_param = next(model.parameters())
                        model_weight_dtype = first_param.dtype
                    
                    # Convert to match model weight dtype
                    if hidden_states.dtype != model_weight_dtype:
                        log_stage(f"Converting hidden_states dtype from {hidden_states.dtype} to {model_weight_dtype} to match model weights")
                        hidden_states = hidden_states.to(dtype=model_weight_dtype)
                    if residual is not None and residual.dtype != model_weight_dtype:
                        log_stage(f"Converting residual dtype from {residual.dtype} to {model_weight_dtype} to match model weights")
                        residual = residual.to(dtype=model_weight_dtype)
                    
                    # Use global decode position from previous stage
                    if global_decode_positions is not None:
                        decode_position = global_decode_positions.to(hidden_states.device)
                        log_stage(f"Using global decode position from previous stage: {decode_position.tolist()}")
                    else:
                        # Fallback: calculate from local (should not happen)
                        log_stage("WARNING: No global decode position received, calculating from local (may cause rotary emb errors)")
                        current_seq_len = initial_seq_len + generation_step
                        decode_position = torch.tensor([initial_seq_len + generation_step - 1], dtype=torch.long, device=hidden_states.device)
                    
                    # Process through model
                    current_seq_len = initial_seq_len + generation_step
                    
                    new_intermediate_tensors = IntermediateTensors({
                        "hidden_states": hidden_states,
                        "residual": residual
                    })
                    
                    # NOTE: In pipeline mode, we don't set complex attention metadata
                    # because each stage is independent and KV cache management is complex.
                    # The model will handle attention without explicit metadata.
                    
                    with set_current_vllm_config(vllm_config=stage_vllm_config):
                        with set_forward_context(
                            None, stage_vllm_config, num_tokens=1
                        ):
                            new_output = model(
                                input_ids=None,
                                positions=decode_position,
                                intermediate_tensors=new_intermediate_tensors,
                            )
                    
                    # Send output to next stage
                    # Always manually send to ensure it's sent
                    if isinstance(new_output, IntermediateTensors):
                        hidden_states_out = new_output["hidden_states"]
                        residual_out = new_output["residual"] if "residual" in new_output.tensors else None
                        
                        # CRITICAL: Get actual dtype from model weights for sending
                        model_weight_dtype = None
                        if hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > 0:
                            first_layer = model.model.layers[0]
                            if hasattr(first_layer, "self_attn") and hasattr(first_layer.self_attn, "q_proj"):
                                model_weight_dtype = first_layer.self_attn.q_proj.weight.dtype
                            elif hasattr(first_layer, "mlp") and hasattr(first_layer.mlp, "gate_proj"):
                                model_weight_dtype = first_layer.mlp.gate_proj.weight.dtype
                        
                        if model_weight_dtype is None:
                            first_param = next(model.parameters())
                            model_weight_dtype = first_param.dtype
                        
                        # PRINT: 上一层输出时的精度（中间 stage decode）
                        log_stage(f"[OUTPUT-DECODE] Stage {stage_idx} step {generation_step} - Model weight dtype: {model_weight_dtype}")
                        log_stage(f"[OUTPUT-DECODE] Stage {stage_idx} step {generation_step} - hidden_states dtype BEFORE conversion: {hidden_states_out.dtype}, shape: {hidden_states_out.shape}")
                        if residual_out is not None:
                            log_stage(f"[OUTPUT-DECODE] Stage {stage_idx} step {generation_step} - residual dtype BEFORE conversion: {residual_out.dtype}, shape: {residual_out.shape}")
                        
                        # Convert to match model weight dtype before sending
                        if hidden_states_out.dtype != model_weight_dtype:
                            log_stage(f"Converting hidden_states_out dtype from {hidden_states_out.dtype} to {model_weight_dtype} before sending")
                            hidden_states_out = hidden_states_out.to(dtype=model_weight_dtype)
                        if residual_out is not None and residual_out.dtype != model_weight_dtype:
                            log_stage(f"Converting residual_out dtype from {residual_out.dtype} to {model_weight_dtype} before sending")
                            residual_out = residual_out.to(dtype=model_weight_dtype)
                        
                        # PRINT: 上一层输出时的精度（转换后，中间 stage decode）
                        log_stage(f"[OUTPUT-DECODE] Stage {stage_idx} step {generation_step} - hidden_states dtype AFTER conversion: {hidden_states_out.dtype}, shape: {hidden_states_out.shape}")
                        if residual_out is not None:
                            log_stage(f"[OUTPUT-DECODE] Stage {stage_idx} step {generation_step} - residual dtype AFTER conversion: {residual_out.dtype}, shape: {residual_out.shape}")
                        
                        new_tensor_dict = {
                            "hidden_states": hidden_states_out,
                            "residual": residual_out,
                            "global_positions": decode_position,  # Pass global decode position to next stage
                            "is_decode": True,  # Mark as decode step
                        }
                        communicator.send_tensor_dict(new_tensor_dict)
                    else:
                        hidden_states_out = new_output
                        
                        # CRITICAL: Get actual dtype from model weights for sending
                        model_weight_dtype = None
                        if hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > 0:
                            first_layer = model.model.layers[0]
                            if hasattr(first_layer, "self_attn") and hasattr(first_layer.self_attn, "q_proj"):
                                model_weight_dtype = first_layer.self_attn.q_proj.weight.dtype
                            elif hasattr(first_layer, "mlp") and hasattr(first_layer.mlp, "gate_proj"):
                                model_weight_dtype = first_layer.mlp.gate_proj.weight.dtype
                        
                        if model_weight_dtype is None:
                            first_param = next(model.parameters())
                            model_weight_dtype = first_param.dtype
                        
                        # PRINT: 上一层输出时的精度（中间 stage decode，非 IntermediateTensors）
                        log_stage(f"[OUTPUT-DECODE] Stage {stage_idx} step {generation_step} - Model weight dtype: {model_weight_dtype}")
                        log_stage(f"[OUTPUT-DECODE] Stage {stage_idx} step {generation_step} - hidden_states dtype BEFORE conversion: {hidden_states_out.dtype}, shape: {hidden_states_out.shape}")
                        
                        # Convert to match model weight dtype before sending
                        if hidden_states_out.dtype != model_weight_dtype:
                            log_stage(f"Converting hidden_states_out dtype from {hidden_states_out.dtype} to {model_weight_dtype} before sending")
                            hidden_states_out = hidden_states_out.to(dtype=model_weight_dtype)
                        
                        # PRINT: 上一层输出时的精度（转换后）
                        log_stage(f"[OUTPUT-DECODE] Stage {stage_idx} step {generation_step} - hidden_states dtype AFTER conversion: {hidden_states_out.dtype}, shape: {hidden_states_out.shape}")
                        
                        new_tensor_dict = {
                            "hidden_states": hidden_states_out,
                            "global_positions": decode_position,
                            "is_decode": True,
                        }
                        communicator.send_tensor_dict(new_tensor_dict)
                    
                    logger.info(f"Stage {stage_idx} processed and forwarded hidden states (step {generation_step})")
                    log_stage(f"Forwarded hidden states for generation step {generation_step}")
                
            except Exception as e:
                logger.error(f"Stage {stage_idx} error: {e}", exc_info=True)
                import traceback
                traceback.print_exc()
            
            communicator.close()
            
    except Exception as e:
        log_stage(f"FATAL ERROR: {e}")
        logger.error(f"Stage {stage_idx} fatal error: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        if output_queue is not None:
            try:
                output_queue.put(f"ERROR: Stage {stage_idx} failed: {e}")
            except:
                pass
    finally:
        log_stage("Worker process exiting")
        logger.info(f"Stage {stage_idx} worker process exiting")


def test_pipeline_engine(
    pipeline_dir: str,
    num_stages: int,
    test_input: str,
    zmq_ports: Optional[list[int]] = None,
    device: str = "cpu",
    max_new_tokens: int = 256,
):
    """Test pipeline using vLLM engine with ZeroMQ"""
    # Set multiprocessing start method
    if device.startswith("cuda"):
        try:
            multiprocessing.set_start_method('spawn', force=True)
            logger.info("Set multiprocessing start method to 'spawn' for CUDA support")
        except RuntimeError:
            pass
    
    pipeline_dir = Path(pipeline_dir)
    
    logger.info(f"Testing pipeline with vLLM engine from {pipeline_dir}")
    logger.info(f"Number of stages: {num_stages}")
    logger.info(f"Test input: {test_input}")
    logger.info(f"Max new tokens: {max_new_tokens}")
    
    # Setup ZeroMQ ports
    if zmq_ports is None:
        base_port = 5000 + random.randint(0, 2000)
        zmq_ports = [base_port + i for i in range(num_stages - 1)]
    
    # Create queues
    input_queue = multiprocessing.Queue() if num_stages > 0 else None
    output_queue = multiprocessing.Queue() if num_stages > 0 else None
    
    # Create shared distributed init file for torch.distributed rendezvous
    dist_fd, dist_init_path = tempfile.mkstemp(prefix="pipeline_dist_", suffix=".tmp")
    os.close(dist_fd)
    dist_init_path = str(dist_init_path)
    
    # Start all stage processes
    processes = []
    logger.info("Starting all stage processes...")
    for i in range(num_stages):
        p = multiprocessing.Process(
            target=stage_worker_process,
            args=(
                i,
                str(pipeline_dir),
                num_stages,
                zmq_ports,
                device,
                dist_init_path,
                input_queue if i == 0 else None,
                output_queue if i == num_stages - 1 else None,
                max_new_tokens,
            ),
        )
        p.start()
        processes.append(p)
        logger.info(f"Started stage {i} process (PID: {p.pid})")
    
    # Wait for all processes to load
    logger.info("Waiting for all stages to load models...")
    print("Waiting for all stages to load models...", flush=True)
    max_wait_time = 180
    start_time = time.time()
    consecutive_alive_count = 0
    required_consecutive = 3
    
    for i in range(max_wait_time):
        time.sleep(2)
        alive_count = sum(1 for p in processes if p.is_alive())
        elapsed = time.time() - start_time
        
        if alive_count == num_stages:
            consecutive_alive_count += 1
            if consecutive_alive_count >= required_consecutive:
                logger.info(f"All {alive_count} processes are alive and stable, proceeding... ({elapsed:.1f}s)")
                print(f"All {alive_count} processes are alive and stable, proceeding... ({elapsed:.1f}s)", flush=True)
                break
            elif i % 5 == 0:
                logger.info(f"All {alive_count} processes are alive, waiting for models to load... ({elapsed:.1f}s)")
                print(f"All {alive_count} processes are alive, waiting for models to load... ({elapsed:.1f}s)", flush=True)
        else:
            consecutive_alive_count = 0
            dead_pids = [p.pid for p in processes if not p.is_alive()]
            logger.warning(f"Only {alive_count}/{num_stages} processes are alive. Dead PIDs: {dead_pids}")
            print(f"WARNING: Only {alive_count}/{num_stages} processes are alive. Dead PIDs: {dead_pids}", flush=True)
    
    final_alive = sum(1 for p in processes if p.is_alive())
    if final_alive < num_stages:
        logger.error(f"Only {final_alive}/{num_stages} processes survived the loading phase")
        print(f"ERROR: Only {final_alive}/{num_stages} processes survived the loading phase", flush=True)
        raise RuntimeError(f"Only {final_alive}/{num_stages} processes survived the loading phase")
    
    logger.info("All processes started. Sending input to stage 0...")
    print("All processes started. Sending input to stage 0...", flush=True)
    
    try:
        print(f"Sending input to stage 0: {test_input}", flush=True)
        input_queue.put(test_input)
        logger.info("Input sent to stage 0, waiting for output from last stage...")
        print("Input sent to stage 0, waiting for output from last stage...", flush=True)
        
        try:
            output = output_queue.get(timeout=300)
            logger.info(f"Inference output: {output}")
            print(f"Inference output: {output}", flush=True)
            return output
        except Exception as queue_error:
            logger.error(f"Failed to get output from queue: {queue_error}", exc_info=True)
            print(f"ERROR: Failed to get output from queue: {queue_error}", flush=True)
            raise
        
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        print(f"ERROR: Inference failed: {e}", flush=True)
        raise
    finally:
        # Send end signal
        if num_stages > 0:
            input_queue.put(None)
        
        # Wait for all processes to end
        for p in processes:
            p.join(timeout=10)
            if p.is_alive():
                logger.warning(f"Process {p.pid} did not terminate, killing...")
                p.terminate()
                p.join()
        try:
            if os.path.exists(dist_init_path):
                os.remove(dist_init_path)
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Test offline pipeline with vLLM engine and ZeroMQ"
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
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate (default: 256)",
    )
    
    args = parser.parse_args()
    
    # Parse ports
    zmq_ports = None
    if args.zmq_ports:
        zmq_ports = [int(p) for p in args.zmq_ports.split(",")]
    
    print(f"Starting pipeline test (vLLM engine version):", flush=True)
    print(f"  Pipeline dir: {args.pipeline_dir}", flush=True)
    print(f"  Num stages: {args.num_stages}", flush=True)
    print(f"  Test input: {args.test_input}", flush=True)
    print(f"  Device: {args.device}", flush=True)
    print(f"  Max new tokens: {args.max_new_tokens}", flush=True)
    
    try:
        output = test_pipeline_engine(
            pipeline_dir=args.pipeline_dir,
            num_stages=args.num_stages,
            test_input=args.test_input,
            zmq_ports=zmq_ports,
            device=args.device,
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
