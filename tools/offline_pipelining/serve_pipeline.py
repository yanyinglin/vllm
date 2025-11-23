#!/usr/bin/env python3
"""
vLLM离线流水线导出工具

将模型按pipeline阶段分割并导出为独立的HuggingFace格式目录。
每个阶段的参数导出为一个独立的文件。
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from vllm.distributed.utils import get_pp_indices
from vllm.logger import init_logger

logger = init_logger(__name__)


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
) -> Dict[str, torch.Tensor]:
    """提取指定pipeline stage的权重（保持原始dtype和device）"""
    start_layer, end_layer = get_stage_layers(num_hidden_layers, pp_rank, pp_size)
    is_first_rank = pp_rank == 0
    is_last_rank = pp_rank == pp_size - 1

    stage_weights = {}
    full_state_dict = model.state_dict()

    # 提取embedding层（第一阶段）
    if is_first_rank:
        for key, value in full_state_dict.items():
            # 匹配embedding相关的键
            if any(
                embed_key in key
                for embed_key in ["embed_tokens", "wte", "word_embeddings"]
            ):
                # 保持原始键名格式（不修改，让HuggingFace自己处理）
                # 这样from_pretrained可以正确匹配
                stage_weights[key] = value.clone()

    # 提取transformer层
    for key, value in full_state_dict.items():
        # 检查是否是transformer层（layers或h）
        if ".layers." in key or ".h." in key or ".transformer." in key:
            # 提取层索引
            parts = key.split(".")
            layer_idx = None
            for i, part in enumerate(parts):
                if part in ["layers", "h"] and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        break
                    except ValueError:
                        continue
                # 对于transformer.h.0.xxx格式
                if part == "transformer" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        break
                    except ValueError:
                        continue

            if layer_idx is not None and start_layer <= layer_idx < end_layer:
                # 重新映射层索引：将原始层索引映射到stage内的相对索引
                # 例如：如果stage包含层8-15，那么层8应该映射为层0
                relative_layer_idx = layer_idx - start_layer
                
                # 重建键名，使用相对层索引
                # 例如：model.layers.8.xxx -> model.layers.0.xxx
                new_parts = []
                replaced = False
                for i, part in enumerate(parts):
                    # 检查是否是层索引部分
                    if i > 0 and parts[i-1] in ["layers", "h"]:
                        try:
                            # 如果当前部分是数字且等于原始层索引，替换为相对索引
                            if int(part) == layer_idx:
                                new_parts.append(str(relative_layer_idx))
                                replaced = True
                            else:
                                new_parts.append(part)
                        except ValueError:
                            # 不是数字，保持原样
                            new_parts.append(part)
                    else:
                        new_parts.append(part)
                
                # 如果成功替换了层索引，使用新键名；否则保持原键名
                if replaced:
                    hf_key = ".".join(new_parts)
                else:
                    # 如果替换失败，保持原键名（可能格式不同）
                    hf_key = key
                
                stage_weights[hf_key] = value.clone()

    # 提取norm层和lm_head（最后阶段）
    if is_last_rank:
        for key, value in full_state_dict.items():
            # 匹配norm层（但不包括layers内的norm）
            if (
                ("norm" in key.lower() or "ln_f" in key.lower())
                and ".layers." not in key
                and ".h." not in key
            ):
                # 保持原始键名格式（不修改，让HuggingFace自己处理）
                stage_weights[key] = value.clone()
            # 匹配lm_head
            if any(head_key in key for head_key in ["lm_head", "embed_out", "head"]):
                # 保持原始键名格式（不修改，让HuggingFace自己处理）
                stage_weights[key] = value.clone()

    return stage_weights


def create_stage_config(
    original_config: dict,
    pp_rank: int,
    pp_size: int,
    num_hidden_layers: int,
    torch_dtype: torch.dtype = None,
) -> dict:
    """为每个stage创建修改后的config"""
    stage_config = original_config.copy()
    start_layer, end_layer = get_stage_layers(num_hidden_layers, pp_rank, pp_size)

    # 修改层数
    stage_config["num_hidden_layers"] = end_layer - start_layer

    # 保存dtype信息（使用新的dtype字段，而不是torch_dtype）
    if torch_dtype is not None:
        dtype_str = str(torch_dtype).replace("torch.", "")
        # 同时保存torch_dtype（向后兼容）和dtype（新格式）
        stage_config["torch_dtype"] = dtype_str
        stage_config["dtype"] = dtype_str

    # 添加pipeline相关信息
    stage_config["_pipeline_info"] = {
        "pp_rank": pp_rank,
        "pp_size": pp_size,
        "start_layer": start_layer,
        "end_layer": end_layer,
        "original_num_hidden_layers": num_hidden_layers,
    }

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
            tokenizer_files = [
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
                "vocab.txt",
                "special_tokens_map.json",
            ]
            for tokenizer_file in tokenizer_files:
                src_path = Path(original_model_path) / tokenizer_file
                if src_path.exists():
                    shutil.copy2(src_path, stage_dir / tokenizer_file)


def get_model_name_from_path(model_path: str) -> str:
    """从模型路径提取模型名称"""
    path = Path(model_path)
    # 如果是目录，使用目录名
    if path.is_dir():
        return path.name
    # 如果是HuggingFace模型ID，使用最后一部分
    if "/" in model_path:
        return model_path.split("/")[-1]
    return model_path


def export_pipeline(
    model_path: str,
    output_dir: str = None,
    pipeline_parallel_size: int = None,
    dtype: str = "auto",
):
    """导出模型为离线流水线"""
    # 如果没有指定output_dir，使用默认路径格式
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
    ) or getattr(original_config, "n_layer", None)
    if num_hidden_layers is None:
        raise ValueError("Cannot determine number of hidden layers from config")

    logger.info(f"Model has {num_hidden_layers} hidden layers")

    # 检测原始模型的dtype
    if dtype == "auto":
        # 尝试从config中获取dtype
        config_dtype = getattr(original_config, "torch_dtype", None)
        if config_dtype is None:
            # 尝试从其他字段推断
            if hasattr(original_config, "dtype"):
                config_dtype = original_config.dtype
        
        if config_dtype is not None:
            if isinstance(config_dtype, str):
                # 字符串转torch dtype
                dtype_map = {
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "float32": torch.float32,
                    "fp16": torch.float16,
                    "bf16": torch.bfloat16,
                    "fp32": torch.float32,
                }
                torch_dtype = dtype_map.get(config_dtype.lower(), torch.float16)
            else:
                torch_dtype = config_dtype
            logger.info(f"Detected dtype from config: {torch_dtype}")
        else:
            # 默认使用float16（大多数LLM模型使用fp16）
            torch_dtype = torch.float16
            logger.info("Using default dtype: float16 (config dtype not found)")
    elif dtype == "float16" or dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16" or dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "float32" or dtype == "fp32":
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float16
        logger.warning(f"Unknown dtype {dtype}, using float16")

    # 检查GPU可用性
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device("cuda:0")
        logger.info("Using GPU for model loading and weight extraction")
    else:
        device = torch.device("cpu")
        logger.warning("GPU not available, using CPU (may convert to fp32)")

    # 加载完整模型（仅加载一次）
    logger.info(f"Loading full model with dtype {torch_dtype}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    
    # 手动将模型移到GPU（如果可用）
    if use_gpu:
        logger.info("Moving model to GPU...")
        model = model.to(device)
    
    model.eval()  # 设置为评估模式

    # 为每个stage导出模型
    for pp_rank in range(pipeline_parallel_size):
        logger.info(
            f"Processing pipeline stage {pp_rank}/{pipeline_parallel_size - 1}"
        )

        # 提取该stage的权重
        stage_weights = extract_stage_weights(
            model=model,
            pp_rank=pp_rank,
            pp_size=pipeline_parallel_size,
            num_hidden_layers=num_hidden_layers,
        )

        if not stage_weights:
            logger.warning(f"No weights extracted for stage {pp_rank}")
            continue

        logger.info(
            f"Extracted {len(stage_weights)} weight tensors for stage {pp_rank}"
        )

        # 创建stage配置
        stage_config_dict = original_config.to_dict()
        stage_config_dict["_original_model_path"] = model_path
        stage_config = create_stage_config(
            original_config=stage_config_dict,
            pp_rank=pp_rank,
            pp_size=pipeline_parallel_size,
            num_hidden_layers=num_hidden_layers,
            torch_dtype=torch_dtype,
        )

        # 将权重移到CPU（保存时需要），但保持dtype
        stage_weights_cpu = {}
        dtype_info = {}
        for key, value in stage_weights.items():
            # 保持原始dtype，只改变device
            original_dtype = value.dtype
            if value.is_cuda:
                stage_weights_cpu[key] = value.cpu()
            else:
                stage_weights_cpu[key] = value
            # 验证dtype是否保持
            if stage_weights_cpu[key].dtype != original_dtype:
                logger.warning(
                    f"Stage {pp_rank} weight {key}: dtype changed from {original_dtype} to {stage_weights_cpu[key].dtype}"
                )
            # 记录dtype信息
            dtype_info[key] = str(original_dtype)
        
        # 记录dtype统计
        unique_dtypes = set(dtype_info.values())
        logger.info(
            f"Stage {pp_rank}: extracted {len(stage_weights_cpu)} weights with dtypes: {unique_dtypes}"
        )

        # 保存为HuggingFace格式
        save_stage_to_hf_format(
            stage_weights=stage_weights_cpu,
            stage_config=stage_config,
            output_dir=output_path,
            stage_idx=pp_rank,
            use_safetensors=True,
        )

    # 清理模型以释放内存
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    logger.info(f"Pipeline export completed. Stages saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export vLLM model as offline pipeline stages"
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
        help="Output directory for pipeline stages (default: /home/yanying/pipeline_export/{model_name})",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Model dtype (default: auto)",
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

