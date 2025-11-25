#!/usr/bin/env python3
"""
vLLM离线流水线导出工具 (修复版)

关键修复：
1. 正确处理模型 dtype，保持原始精度
2. 默认在 GPU 上进行拆分（如果可用）
3. 修复 pipeline 拆分逻辑，确保权重正确分配
4. 优化层索引处理，支持多种模型架构
5. 修复首尾 stage 权重提取问题
6. 修复 create_stage_config 参数缺失问题
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
)

from vllm.distributed.utils import get_pp_indices
from vllm.logger import init_logger

logger = init_logger(__name__)

# 支持的模型类型及其键名模式
MODEL_KEY_PATTERNS = {
    "llama": {
        "layer_pattern": r"model\.layers\.(\d+)",
        "embed_key": "model.embed_tokens",
        "norm_key": "model.norm",
        "lm_head_key": "lm_head",
        "layer_prefix": "model.layers.",
    },
    "gpt2": {
        "layer_pattern": r"transformer\.h\.(\d+)",
        "embed_key": "transformer.wte",
        "norm_key": "transformer.ln_f",
        "lm_head_key": "lm_head",
        "layer_prefix": "transformer.h.",
    },
    "falcon": {
        "layer_pattern": r"transformer\.h\.(\d+)",
        "embed_key": "transformer.word_embeddings",
        "norm_key": "transformer.ln_f",
        "lm_head_key": "lm_head",
        "layer_prefix": "transformer.h.",
    },
    "gptj": {
        "layer_pattern": r"transformer\.h\.(\d+)",
        "embed_key": "transformer.wte",
        "norm_key": "transformer.ln_f",
        "lm_head_key": "lm_head",
        "layer_prefix": "transformer.h.",
    },
    "mpt": {
        "layer_pattern": r"transformer\.blocks\.(\d+)",
        "embed_key": "transformer.wte",
        "norm_key": "transformer.norm_f",
        "lm_head_key": "lm_head",
        "layer_prefix": "transformer.blocks.",
    },
    "opt": {
        "layer_pattern": r"model\.decoder\.layers\.(\d+)",
        "embed_key": "model.decoder.embed_tokens",
        "norm_key": "model.decoder.final_layer_norm",
        "lm_head_key": "lm_head",
        "layer_prefix": "model.decoder.layers.",
    },
    "bloom": {
        "layer_pattern": r"transformer\.h\.(\d+)",
        "embed_key": "transformer.word_embeddings",
        "norm_key": "transformer.ln_f",
        "lm_head_key": "lm_head",
        "layer_prefix": "transformer.h.",
    },
    "qwen": {
        "layer_pattern": r"transformer\.h\.(\d+)",
        "embed_key": "transformer.wte",
        "norm_key": "transformer.ln_f",
        "lm_head_key": "lm_head",
        "layer_prefix": "transformer.h.",
    },
    "chatglm": {
        "layer_pattern": r"transformer\.encoder\.layers\.(\d+)",
        "embed_key": "transformer.embedding.word_embeddings",
        "norm_key": "transformer.encoder.final_layernorm",
        "lm_head_key": "transformer.output_layer",
        "layer_prefix": "transformer.encoder.layers.",
    },
}

# 位置编码相关键（需要特殊处理）
ROPE_KEYS = [
    "rotary_emb.inv_freq",  # Llama 格式
    "model.layers.*.self_attn.rotary_emb.inv_freq",  # Llama 变体
    "transformer.h.*.attn.rotary_emb.inv_freq",  # GPT-NeoX 格式
    "transformer.blocks.*.norm_1.rotary_emb.inv_freq",  # MPT 格式
]

def get_model_patterns(config: PretrainedConfig) -> dict:
    """根据模型类型获取键名模式"""
    model_type = getattr(config, "model_type", "").lower()
    
    # 尝试匹配已知架构
    for key, patterns in MODEL_KEY_PATTERNS.items():
        if key in model_type:
            return patterns
    
    # 特殊处理 Qwen
    if "qwen" in model_type:
        return MODEL_KEY_PATTERNS["qwen"]
    
    # 特殊处理 ChatGLM
    if "chatglm" in model_type:
        return MODEL_KEY_PATTERNS["chatglm"]
    
    # 默认回退到Llama模式（最常见）
    logger.warning(
        f"Unknown model type: {model_type}. Using default Llama patterns. "
        "You may need to add custom patterns in MODEL_KEY_PATTERNS."
    )
    return MODEL_KEY_PATTERNS["llama"]


def get_stage_layers(
    num_hidden_layers: int,
    pp_rank: int,
    pp_size: int,
) -> Tuple[int, int]:
    """获取指定pipeline stage的层范围"""
    return get_pp_indices(num_hidden_layers, pp_rank, pp_size)


def extract_stage_weights(
    model: nn.Module,
    pp_rank: int,
    pp_size: int,
    num_hidden_layers: int,
    model_patterns: dict,
) -> Tuple[Dict[str, torch.Tensor], Dict[int, int]]:
    """
    提取指定pipeline stage的权重（保持原始dtype和device）
    
    Args:
        model: 完整模型
        pp_rank: 当前stage的rank
        pp_size: 总stage数
        num_hidden_layers: 模型总层数
        model_patterns: 模型键名模式（来自get_model_patterns）
    
    Returns:
        stage_weights: 该stage的权重字典
        global_layer_map: 映射 {local_idx: global_idx}，仅包含transformer层
    """
    start_layer, end_layer = get_stage_layers(num_hidden_layers, pp_rank, pp_size)
    is_first_rank = pp_rank == 0
    is_last_rank = pp_rank == pp_size - 1

    logger.info(
        f"Stage {pp_rank}: Extracting layers {start_layer} to {end_layer-1} "
        f"(inclusive, total layers: {num_hidden_layers})"
    )

    stage_weights = {}
    global_layer_map = {}
    full_state_dict = model.state_dict()
    
    # 记录所有原始键
    all_original_keys = set(full_state_dict.keys())
    extracted_keys = set()
    layer_pattern = re.compile(model_patterns["layer_pattern"])
    layer_prefix = model_patterns["layer_prefix"]

    # 1. 提取embedding层（仅首阶段）
    if is_first_rank:
        embed_key = model_patterns["embed_key"]
        for key in list(full_state_dict.keys()):
            # 精确匹配embedding键（可能有多个变体）
            if (embed_key in key or
                ("embed" in key.lower() and
                 ("token" in key.lower() or "wte" in key.lower() or "word_embeddings" in key.lower())) or
                key == "model.embed_tokens.weight"):
                # 排除lm_head（即使它与embedding共享权重）
                if "lm_head" not in key and "output_layer" not in key:
                    stage_weights[key] = full_state_dict[key].clone()
                    extracted_keys.add(key)
                    logger.debug(f"Stage {pp_rank}: Extracted embedding: {key}")
    else:
        logger.debug(f"Stage {pp_rank} is not first stage, skipping embedding layers")

    # 2. 提取transformer层
    for key, value in full_state_dict.items():
        # 跳过运行时缓存（cos/sin缓存）
        if any(skip in key for skip in ["cos_cached", "sin_cached"]):
            continue
            
        # 保留rope的inv_freq（用于重建位置编码）
        if "rotary_emb.inv_freq" in key or key.endswith("rotary_emb.inv_freq"):
            stage_weights[key] = value.clone()
            extracted_keys.add(key)
            continue
        
        # 检查是否是transformer层
        match = layer_pattern.search(key)
        if not match:
            continue
            
        try:
            layer_idx = int(match.group(1))
        except (ValueError, IndexError):
            logger.warning(f"Failed to parse layer index from key: {key}")
            continue
        
        # 检查层索引是否在当前stage范围内
        if start_layer <= layer_idx < end_layer:
            relative_layer_idx = layer_idx - start_layer
            global_layer_map[relative_layer_idx] = layer_idx
            
            # 精确替换层索引（只替换匹配到的第一个数字）
            new_key = layer_pattern.sub(
                f"{layer_prefix}{relative_layer_idx}", 
                key,
                count=1
            )
            
            stage_weights[new_key] = value.clone()
            extracted_keys.add(key)
            logger.debug(
                f"Stage {pp_rank}: Remapped layer {layer_idx} -> {relative_layer_idx}: "
                f"{key} -> {new_key}"
            )

    # 3. 提取norm层和lm_head（仅尾阶段）
    if is_last_rank:
        # 提取final norm - more specific regex pattern matching
        norm_key = model_patterns["norm_key"]
        for key in list(full_state_dict.keys()):
            # Match exact norm key patterns for the model
            if norm_key in key or (
                "norm.weight" in key and
                ("model.norm" in key or "transformer.ln_f" in key or "transformer.norm_f" in key)
            ):
                # 确保不是层内的norm (not in transformer layers)
                if not any(prefix in key for prefix in [
                    "layers.", "h.", "blocks.", "decoder.layers.", "encoder.layers."
                ]) and not re.search(r"\.\d+\.", key):  # avoid layer-specific norms
                    stage_weights[key] = full_state_dict[key].clone()
                    extracted_keys.add(key)
                    logger.debug(f"Stage {pp_rank}: Extracted final norm: {key}")

        # 提取lm_head - more specific matching
        lm_head_key = model_patterns["lm_head_key"]
        for key in list(full_state_dict.keys()):
            # More robust matching for lm_head
            if ("lm_head" in key and "weight" in key) or key == "lm_head.weight":
                stage_weights[key] = full_state_dict[key].clone()
                extracted_keys.add(key)
                logger.debug(f"Stage {pp_rank}: Extracted lm_head: {key}")
    else:
        logger.debug(f"Stage {pp_rank} is not last stage, skipping lm_head and final norm")

    # 4. 检查遗漏的关键权重
    missing_critical = []
    if is_first_rank and not any("embed_tokens.weight" in k or "wte.weight" in k or "word_embeddings.weight" in k for k in stage_weights):
        missing_critical.append("embedding layer")
    if is_last_rank and not any("norm.weight" in k or "ln_f.weight" in k or "norm_f.weight" in k for k in stage_weights):
        missing_critical.append("final norm")
    if is_last_rank and not any("lm_head.weight" in k for k in stage_weights):
        missing_critical.append("lm_head")

    if missing_critical:
        logger.warning(
            f"Stage {pp_rank} is missing critical components: {', '.join(missing_critical)}. "
            "This may cause inference failures."
        )

    # 5. 统计报告
    logger.info(
        f"Stage {pp_rank}: Extracted {len(extracted_keys)}/{len(all_original_keys)} keys. "
        f"Global layer map: {len(global_layer_map)} layers"
    )
    
    return stage_weights, global_layer_map


def create_stage_config(
    original_config: dict,
    pp_rank: int,
    pp_size: int,
    num_hidden_layers: int,
    torch_dtype: torch.dtype,
    global_layer_map: Dict[int, int],
    model_patterns: dict,
) -> dict:
    """为每个stage创建修改后的config，移除不属于当前stage的组件"""
    stage_config = original_config.copy()
    start_layer, end_layer = get_stage_layers(num_hidden_layers, pp_rank, pp_size)
    is_first_rank = pp_rank == 0
    is_last_rank = pp_rank == pp_size - 1

    # 修改层数
    stage_config["num_hidden_layers"] = end_layer - start_layer

    # 保存dtype信息
    dtype_str = str(torch_dtype).replace("torch.", "")
    stage_config["torch_dtype"] = dtype_str
    stage_config["dtype"] = dtype_str

    # 添加pipeline元数据
    stage_config["_pipeline_info"] = {
        "pp_rank": pp_rank,
        "pp_size": pp_size,
        "start_layer": start_layer,
        "end_layer": end_layer,
        "original_num_hidden_layers": num_hidden_layers,
        "global_layer_map": global_layer_map,
        "is_first_stage": is_first_rank,
        "is_last_stage": is_last_rank,
    }

    # === 关键修复：根据stage类型移除不必要的配置 ===
    model_type = stage_config.get("model_type", "").lower()

    # 非最后stage：移除lm_head相关配置
    if not is_last_rank:
        # 禁用权重共享（embedding和lm_head共享权重）
        if "tie_word_embeddings" in stage_config:
            stage_config["tie_word_embeddings"] = False

        # 添加标记表示没有lm_head - this tells the model not to expect lm_head weights
        stage_config["_skip_lm_head"] = True
        # For HuggingFace compatibility, explicitly set lm_head related configs to None/False if they exist
        if "num_labels" in stage_config:
            stage_config["num_labels"] = None
        # For models that might try to initialize lm_head
        if "_name_or_path" in stage_config:
            # Don't modify name to avoid confusion
            pass

    # 非最后stage：移除final norm configuration
    if not is_last_rank:
        # 添加标记表示没有final norm
        stage_config["_no_final_norm"] = True
        # Remove norm-specific parameters that shouldn't exist in intermediate stages
        if "rms_norm_eps" in stage_config and not is_last_rank:
            # Keep the parameter but mark that final norm doesn't exist
            stage_config["_has_final_norm"] = False

    # 非首stage：移除embedding configuration
    if not is_first_rank:
        stage_config["_no_embedding"] = True
        # Mark that this stage doesn't have embeddings
        stage_config["_has_embeddings"] = False

    # 针对Llama的特殊处理
    if "llama" in model_type:
        if not is_last_rank:
            # Explicitly mark that this Llama stage doesn't have lm_head
            stage_config["_no_lm_head"] = True
        if not is_first_rank:
            # Mark that this Llama stage doesn't have embed_tokens
            stage_config["_no_embed_tokens"] = True

    # 为vLLM添加特殊标记
    stage_config["_is_pipeline_stage"] = True
    stage_config["_pp_rank"] = pp_rank
    stage_config["_pp_size"] = pp_size

    return stage_config


def save_stage_to_hf_format(
    stage_weights: Dict[str, torch.Tensor],
    stage_config: dict,
    output_dir: Path,
    stage_idx: int,
    use_safetensors: bool = True,
):
    """将stage保存为HuggingFace格式"""
    stage_dir = output_dir / f"stage_{stage_idx}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    # 保存config.json
    config_path = stage_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(stage_config, f, indent=2, ensure_ascii=False)
    
    # 保存pipeline stage标记文件
    with open(stage_dir / ".pipeline_stage", "w") as f:
        f.write(f"pp_rank={stage_idx}\n")
        f.write(f"pp_size={stage_config['_pipeline_info']['pp_size']}\n")
        f.write(f"is_first_stage={stage_config['_pipeline_info']['is_first_stage']}\n")
        f.write(f"is_last_stage={stage_config['_pipeline_info']['is_last_stage']}\n")

    # 保存权重
    if use_safetensors:
        try:
            from safetensors.torch import save_file

            weights_path = stage_dir / "model.safetensors"
            save_file(stage_weights, weights_path)
            logger.info(f"Saved stage {stage_idx} weights to {weights_path}")
        except ImportError:
            logger.warning("safetensors not available, using pytorch format")
            use_safetensors = False

    if not use_safetensors:
        weights_path = stage_dir / "pytorch_model.bin"
        torch.save(stage_weights, weights_path)
        logger.info(f"Saved stage {stage_idx} weights to {weights_path}")

    # 复制tokenizer文件（仅第一阶段）
    if stage_idx == 0:
        original_model_path = stage_config.get("_original_model_path", "")
        if original_model_path and os.path.exists(original_model_path):
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    original_model_path, 
                    trust_remote_code=True,
                    use_fast=False  # 使用慢速tokenizer确保兼容性
                )
                tokenizer.save_pretrained(stage_dir)
                logger.info(f"Copied tokenizer to {stage_dir}")
            except Exception as e:
                logger.warning(f"Failed to copy tokenizer: {str(e)}")
                # 尝试复制原始文件
                tokenizer_files = [
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "vocab.json",
                    "merges.txt",
                    "vocab.txt",
                    "special_tokens_map.json",
                    "tokenizer.model",  # SentencePiece
                ]
                for tokenizer_file in tokenizer_files:
                    src_path = Path(original_model_path) / tokenizer_file
                    if src_path.exists():
                        shutil.copy2(src_path, stage_dir / tokenizer_file)
                        logger.info(f"Copied {tokenizer_file} to {stage_dir}")


def get_model_name_from_path(model_path: str) -> str:
    """从模型路径提取模型名称"""
    path = Path(model_path)
    if path.is_dir():
        return path.name
    if "/" in model_path:
        return model_path.split("/")[-1]
    return model_path


def detect_model_dtype(config: PretrainedConfig) -> torch.dtype:
    """从配置中检测模型的原始 dtype"""
    # 1. 优先从 config 中获取
    config_dtype = getattr(config, "torch_dtype", None)
    if config_dtype is None:
        config_dtype = getattr(config, "dtype", None)
    
    if config_dtype is not None:
        if isinstance(config_dtype, str):
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "fp32": torch.float32,
            }
            return dtype_map.get(config_dtype.lower(), torch.float16)
        elif isinstance(config_dtype, torch.dtype):
            return config_dtype
    
    # 2. 检查是否有 fp16/bf16 相关字段
    if getattr(config, "fp16", False) or getattr(config, "half_precision", False):
        return torch.float16
    if getattr(config, "bf16", False):
        return torch.bfloat16
    
    # 3. 默认使用 float16（大多数LLM使用）
    return torch.float16


def export_pipeline(
    model_path: str,
    output_dir: str = None,
    pipeline_parallel_size: int = None,
    dtype: str = "auto",
):
    """导出模型为离线流水线"""
    # 设置输出目录
    if output_dir is None:
        model_name = get_model_name_from_path(model_path)
        output_dir = f"/home/yanying/pipeline_export/{model_name}"
        logger.info(f"Using default output directory: {output_dir}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model from {model_path}")
    logger.info(f"Pipeline parallel size: {pipeline_parallel_size}")
    logger.info(f"Output directory: {output_path}")

    # 加载原始配置
    original_config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=True
    )
    num_hidden_layers = getattr(
        original_config, "num_hidden_layers", None
    ) or getattr(original_config, "n_layer", None) or getattr(original_config, "num_layers", None)
    
    if num_hidden_layers is None:
        raise ValueError("Cannot determine number of hidden layers from config")

    logger.info(f"Model has {num_hidden_layers} hidden layers")
    model_patterns = get_model_patterns(original_config)
    logger.info(f"Using model patterns for: {model_patterns}")

    # 确定dtype - 修复bug：正确检测原始dtype
    if dtype == "auto":
        torch_dtype = detect_model_dtype(original_config)
        logger.info(f"Auto-detected model dtype: {torch_dtype}")
    else:
        dtype_map = {
            "float16": torch.float16, "fp16": torch.float16,
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float32": torch.float32, "fp32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype.lower(), torch.float16)
        logger.info(f"Using specified dtype: {torch_dtype}")

    # 检查GPU
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")
    logger.info(f"Using device: {device} (GPU available: {use_gpu})")
    
    # 如果使用GPU，检查内存
    if use_gpu:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        logger.info(f"GPU memory available: {gpu_memory:.2f} GB")
        
        # 估计模型内存需求（粗略估计）
        param_count = getattr(original_config, "num_parameters", 0)
        if param_count == 0:
            # 尝试估计参数量
            hidden_size = getattr(original_config, "hidden_size", 4096)
            num_layers = num_hidden_layers
            vocab_size = getattr(original_config, "vocab_size", 32000)
            param_count = (hidden_size * hidden_size * num_layers * 12) + (vocab_size * hidden_size)
        
        # 估计内存需求 (float16: 2 bytes/param, float32: 4 bytes/param)
        bytes_per_param = 2 if torch_dtype == torch.float16 else 4
        estimated_memory_gb = (param_count * bytes_per_param) / (1024**3)
        logger.info(f"Estimated model memory requirement: {estimated_memory_gb:.2f} GB")
        
        if estimated_memory_gb > gpu_memory * 0.8:
            logger.warning(
                f"Estimated memory requirement ({estimated_memory_gb:.2f} GB) exceeds 80% of available GPU memory ({gpu_memory:.2f} GB). "
                "Consider using CPU or a smaller model."
            )

    # === 关键修复：确保在GPU上加载并保持原始精度 ===
    logger.info(f"Loading full model with dtype {torch_dtype}...")
    load_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    
    # 明确指定torch_dtype
    if torch_dtype == torch.float16:
        load_kwargs["torch_dtype"] = torch.float16
    elif torch_dtype == torch.bfloat16:
        load_kwargs["torch_dtype"] = torch.bfloat16
    elif torch_dtype == torch.float32:
        load_kwargs["torch_dtype"] = torch.float32
    
    # 优先使用GPU
    if use_gpu:
        load_kwargs["device_map"] = "auto" if pipeline_parallel_size > 1 else "cuda:0"
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

    # 验证dtype - 修复bug：确保保持原始精度
    first_param_dtype = next(model.parameters()).dtype
    if first_param_dtype != torch_dtype:
        logger.warning(
            f"Model dtype ({first_param_dtype}) doesn't match requested dtype ({torch_dtype}). "
            "This may indicate a loading issue. Attempting to convert..."
        )
        # 只在必要时转换
        model = model.to(torch_dtype)
        first_param_dtype = next(model.parameters()).dtype
        if first_param_dtype != torch_dtype:
            logger.error(
                f"Failed to convert model to {torch_dtype}. Current dtype: {first_param_dtype}. "
                "This will likely cause issues during export."
            )
    
    # 验证所有参数的dtype
    all_dtypes = set(p.dtype for p in model.parameters())
    if len(all_dtypes) > 1:
        logger.warning(f"Model has mixed dtypes: {all_dtypes}. This may cause issues.")
    
    model.eval()

    # 为每个stage导出
    for pp_rank in range(pipeline_parallel_size):
        logger.info(f"Processing pipeline stage {pp_rank}/{pipeline_parallel_size-1}")

        # 提取权重
        stage_weights, global_layer_map = extract_stage_weights(
            model=model,
            pp_rank=pp_rank,
            pp_size=pipeline_parallel_size,
            num_hidden_layers=num_hidden_layers,
            model_patterns=model_patterns,
        )

        if not stage_weights:
            raise ValueError(f"No weights extracted for stage {pp_rank}")

        logger.info(
            f"Stage {pp_rank}: Extracted {len(stage_weights)} weights. "
            f"Layer map: {len(global_layer_map)} layers"
        )

        # 移动到CPU并验证dtype - 修复bug：保持原始dtype
        stage_weights_cpu = {}
        dtype_mismatches = 0
        for key, value in stage_weights.items():
            # 保持原始dtype，仅移动到CPU
            cpu_value = value.cpu() if value.is_cuda else value
            
            # 验证dtype
            if cpu_value.dtype != torch_dtype:
                logger.warning(
                    f"Weight {key} has dtype {cpu_value.dtype}, expected {torch_dtype}. "
                    "This may indicate a model loading issue."
                )
                dtype_mismatches += 1
            
            stage_weights_cpu[key] = cpu_value

        if dtype_mismatches > 0:
            logger.warning(
                f"Stage {pp_rank} has {dtype_mismatches} weights with unexpected dtype. "
                "This may cause issues during inference."
            )

        # 创建config - 修复：添加缺失的model_patterns参数
        stage_config_dict = original_config.to_dict()
        stage_config_dict["_original_model_path"] = model_path
        stage_config = create_stage_config(
            original_config=stage_config_dict,
            pp_rank=pp_rank,
            pp_size=pipeline_parallel_size,
            num_hidden_layers=num_hidden_layers,
            torch_dtype=torch_dtype,
            global_layer_map=global_layer_map,
            model_patterns=model_patterns,  # 修复：添加缺失的参数
        )

        # 保存
        save_stage_to_hf_format(
            stage_weights=stage_weights_cpu,
            stage_config=stage_config,
            output_dir=output_path,
            stage_idx=pp_rank,
            use_safetensors=True,
        )

    # 清理
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Pipeline export completed successfully. Stages saved to {output_path}")

    # 验证导出结果
    logger.info("Verifying exported stages...")
    for pp_rank in range(pipeline_parallel_size):
        stage_dir = output_path / f"stage_{pp_rank}"
        config_path = stage_dir / "config.json"
        weights_path = stage_dir / "model.safetensors"
        
        if not config_path.exists():
            logger.error(f"Config file missing for stage {pp_rank}: {config_path}")
        if not weights_path.exists():
            logger.error(f"Weights file missing for stage {pp_rank}: {weights_path}")
        
        # 检查权重键
        if weights_path.exists():
            try:
                from safetensors import safe_open
                with safe_open(weights_path, framework="pt") as f:
                    weight_keys = list(f.keys())
                    logger.info(f"Stage {pp_rank} has {len(weight_keys)} weight tensors")
                    
                    # 验证首尾stage
                    is_first = pp_rank == 0
                    is_last = pp_rank == pipeline_parallel_size - 1
                    
                    has_embedding = any("embed" in k for k in weight_keys)
                    has_lm_head = any("lm_head" in k or "output_layer" in k for k in weight_keys)
                    
                    if is_first and not has_embedding:
                        logger.warning(f"Stage {pp_rank} is first stage but has no embedding weights")
                    if is_last and not has_lm_head:
                        logger.warning(f"Stage {pp_rank} is last stage but has no lm_head weights")
                    if not is_last and has_lm_head:
                        logger.warning(f"Stage {pp_rank} is not last stage but has lm_head weights")
            except Exception as e:
                logger.warning(f"Failed to verify weights for stage {pp_rank}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description="Export vLLM model as offline pipeline stages (Fixed Version)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model or HuggingFace model ID",
    )
    parser.add_argument(
        "--pipeline-parallel-size",
        type=int,
        required=True,
        help="Number of pipeline parallel stages",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: /home/yanying/pipeline_export/{model_name})",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32", "fp16", "bf16", "fp32"],
        help="Model dtype (default: auto, detects original dtype)",
    )

    args = parser.parse_args()

    export_pipeline(
        model_path=args.model,
        output_dir=args.output_dir,
        pipeline_parallel_size=args.pipeline_parallel_size,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()