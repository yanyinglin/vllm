#!/usr/bin/env python3
"""
vLLM pipeline stage tester.

Discovers exported `stage_*` folders, validates their coverage, and invokes
the vLLM inference engine end-to-end (no ZeroMQ) to check generation quality.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig
from vllm.entrypoints.llm import LLM
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams

logger = init_logger(__name__)

STAGE_PATTERN = re.compile(r"^stage_(\d+)$")
DEFAULT_PIPELINE_DIR = "/home/yanying/pipeline_export/Llama-3-8B-final-fix"


def discover_stage_dirs(pipeline_dir: Path) -> List[Tuple[int, Path]]:
    """Return sorted (stage_idx, path) pairs for all stage directories."""
    stage_pairs: List[Tuple[int, Path]] = []
    for child in pipeline_dir.iterdir():
        if not child.is_dir():
            continue
        match = STAGE_PATTERN.match(child.name)
        if not match:
            continue
        stage_idx = int(match.group(1))
        stage_pairs.append((stage_idx, child))

    if not stage_pairs:
        raise FileNotFoundError(f"No stage_* directories found under {pipeline_dir}")

    stage_pairs.sort(key=lambda item: item[0])
    expected = list(range(len(stage_pairs)))
    actual = [idx for idx, _ in stage_pairs]
    if actual != expected:
        raise ValueError(
            f"Stage directories must be contiguous starting at 0. "
            f"Found indices: {actual}"
        )
    return stage_pairs


def load_stage_metadata(stage_pairs: Sequence[Tuple[int, Path]]) -> List[Dict[str, object]]:
    """Load minimal metadata (config, dtype, weights) for each stage."""
    metadata: List[Dict[str, object]] = []
    for stage_idx, stage_path in stage_pairs:
        config_path = stage_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config.json for stage {stage_idx}: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        dtype = config.get("dtype") or config.get("torch_dtype") or "float16"
        pipeline_info = config.get("_pipeline_info", {})
        weights_path: Optional[Path] = None
        for candidate in ("model.safetensors", "pytorch_model.bin"):
            candidate_path = stage_path / candidate
            if candidate_path.exists():
                weights_path = candidate_path
                break

        metadata.append(
            {
                "idx": stage_idx,
                "path": stage_path,
                "config": config,
                "dtype": dtype,
                "weights": str(weights_path) if weights_path else None,
                "start_layer": pipeline_info.get("start_layer"),
                "end_layer": pipeline_info.get("end_layer"),
                "is_first": pipeline_info.get("is_first_stage", stage_idx == 0),
                "is_last": pipeline_info.get("is_last_stage", stage_idx == len(stage_pairs) - 1),
            }
        )
    return metadata


def summarize_pipeline(metadata: Sequence[Dict[str, object]], pipeline_dir: Path) -> None:
    logger.info("Detected %d pipeline stages under %s", len(metadata), pipeline_dir)
    for entry in metadata:
        logger.info(
            "Stage %d | layers %s-%s | dtype=%s | weights=%s",
            entry["idx"],
            entry.get("start_layer"),
            entry.get("end_layer"),
            entry.get("dtype"),
            entry.get("weights") or "MISSING",
        )

    stage0_config = metadata[0]["config"] if metadata else {}
    original_path = None
    if isinstance(stage0_config, dict):
        original_path = stage0_config.get("_original_model_path")
    if original_path:
        logger.info("Original model path recorded in config: %s", original_path)


def validate_stage_layout(metadata: Sequence[Dict[str, object]]) -> None:
    coverage = []
    for entry in metadata:
        start = entry.get("start_layer")
        end = entry.get("end_layer")
        if start is not None and end is not None:
            coverage.append((int(start), int(end), entry["idx"]))
    if not coverage:
        logger.warning("Stage configs lack start/end layer info; cannot verify layer coverage.")
        return

    coverage.sort(key=lambda item: item[0])
    expected_start = coverage[0][0]
    for start, end, idx in coverage:
        if start != expected_start:
            logger.warning(
                "Layer gap detected before stage %s: expected start %s, found %s",
                idx,
                expected_start,
                start,
            )
        if end <= start:
            logger.warning("Stage %s end_layer (%s) <= start_layer (%s)", idx, end, start)
        expected_start = end


MODEL_KEY_PATTERNS = {
    "llama": {
        "layer_pattern": r"model\.layers\.(\d+)",
        "layer_prefix": "model.layers.",
    },
    "gpt2": {
        "layer_pattern": r"transformer\.h\.(\d+)",
        "layer_prefix": "transformer.h.",
    },
    "falcon": {
        "layer_pattern": r"transformer\.h\.(\d+)",
        "layer_prefix": "transformer.h.",
    },
    "gptj": {
        "layer_pattern": r"transformer\.h\.(\d+)",
        "layer_prefix": "transformer.h.",
    },
    "mpt": {
        "layer_pattern": r"transformer\.blocks\.(\d+)",
        "layer_prefix": "transformer.blocks.",
    },
    "opt": {
        "layer_pattern": r"model\.decoder\.layers\.(\d+)",
        "layer_prefix": "model.decoder.layers.",
    },
    "bloom": {
        "layer_pattern": r"transformer\.h\.(\d+)",
        "layer_prefix": "transformer.h.",
    },
    "qwen": {
        "layer_pattern": r"transformer\.h\.(\d+)",
        "layer_prefix": "transformer.h.",
    },
    "chatglm": {
        "layer_pattern": r"transformer\.encoder\.layers\.(\d+)",
        "layer_prefix": "transformer.encoder.layers.",
    },
}


def get_model_patterns(config: PretrainedConfig) -> dict:
    model_type = getattr(config, "model_type", "").lower()
    for key, patterns in MODEL_KEY_PATTERNS.items():
        if key in model_type:
            return patterns
    if "qwen" in model_type:
        return MODEL_KEY_PATTERNS["qwen"]
    if "chatglm" in model_type:
        return MODEL_KEY_PATTERNS["chatglm"]
    logger.warning(
        "Unknown model type %s when remapping weights; defaulting to llama-style patterns.",
        model_type,
    )
    return MODEL_KEY_PATTERNS["llama"]


def remap_stage_key(
    key: str,
    stage_info: Dict[str, object],
    patterns: dict,
) -> str:
    layer_pattern = re.compile(patterns["layer_pattern"])
    match = layer_pattern.search(key)
    if not match:
        return key
    local_idx = int(match.group(1))
    global_map = stage_info.get("global_layer_map") or {}
    if isinstance(global_map, dict) and global_map:
        global_idx = int(global_map.get(str(local_idx), global_map.get(local_idx, local_idx)))
    else:
        start_layer = int(stage_info.get("start_layer", 0))
        global_idx = start_layer + local_idx
    return layer_pattern.sub(f"{patterns['layer_prefix']}{global_idx}", key, count=1)


def build_temp_full_model(
    pipeline_dir: Path,
    metadata: Sequence[Dict[str, object]],
) -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="pipeline_vllm_full_"))
    stage0_config_path = metadata[0]["path"] / "config.json"
    stage0_config = AutoConfig.from_pretrained(str(stage0_config_path), trust_remote_code=True)
    original_path = getattr(stage0_config, "_original_model_path", None)
    if original_path:
        base_config = AutoConfig.from_pretrained(original_path, trust_remote_code=True)
    else:
        base_config = stage0_config
    base_config.save_pretrained(tmp_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(metadata[0]["path"]), trust_remote_code=True)
    tokenizer.save_pretrained(tmp_dir)
    patterns = get_model_patterns(base_config)
    weight_map: Dict[str, str] = {}
    total_size = 0
    for entry in metadata:
        stage_idx = entry["idx"]
        stage_dir = entry["path"]
        weights_path = stage_dir / "model.safetensors"
        if not weights_path.exists():
            weights_path = stage_dir / "pytorch_model.bin"
        if not weights_path.exists():
            raise FileNotFoundError(f"No weight file found for stage {stage_idx}: {stage_dir}")
        stage_info = entry["config"].get("_pipeline_info", {})
        shard_name = f"model-stage-{stage_idx:05d}-of-{len(metadata):05d}.safetensors"
        remapped: Dict[str, torch.Tensor] = {}
        if weights_path.suffix == ".safetensors":
            with safe_open(str(weights_path), framework="pt") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    new_key = remap_stage_key(key, stage_info, patterns)
                    remapped[new_key] = tensor.cpu()
                    weight_map[new_key] = shard_name
                    total_size += tensor.numel() * tensor.element_size()
        else:
            state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
            for key, tensor in state.items():
                if not isinstance(tensor, torch.Tensor):
                    continue
                new_key = remap_stage_key(key, stage_info, patterns)
                remapped[new_key] = tensor.cpu()
                weight_map[new_key] = shard_name
                total_size += tensor.numel() * tensor.element_size()
        save_file(remapped, str(tmp_dir / shard_name))
    index_path = tmp_dir / "model.safetensors.index.json"
    index_payload = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    index_path.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")
    return tmp_dir


def run_with_vllm_engine(
    pipeline_dir: Path,
    num_stages: int,
    metadata: Sequence[Dict[str, object]],
    prompt: str,
    max_new_tokens: int,
    dtype: Optional[str],
    log_stats: bool,
    gpu_ids: Optional[List[int]] = None,
) -> Tuple[str, Dict[str, object]]:
    # Set CUDA_VISIBLE_DEVICES if GPU IDs are specified
    old_cuda_visible = None
    if gpu_ids is not None:
        if len(gpu_ids) != num_stages:
            raise ValueError(
                f"Number of GPU IDs ({len(gpu_ids)}) must match number of stages ({num_stages})"
            )
        gpu_str = ",".join(str(gpu_id) for gpu_id in gpu_ids)
        old_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
        logger.info(f"Setting CUDA_VISIBLE_DEVICES={gpu_str} for {num_stages} pipeline stages")
        try:
            # Verify GPU availability
            if torch.cuda.is_available():
                available_count = torch.cuda.device_count()
                if available_count < num_stages:
                    logger.warning(
                        f"Only {available_count} GPU(s) visible, but {num_stages} stages requested. "
                        "Some stages may share GPUs."
                    )
        except Exception as e:
            logger.warning(f"Could not verify GPU availability: {e}")

    temp_model_dir = build_temp_full_model(pipeline_dir, metadata)
    try:
        llm = LLM(
            model=str(temp_model_dir),
            tokenizer=str(temp_model_dir),
            trust_remote_code=True,
            tensor_parallel_size=1,
            dtype=dtype or "auto",
            enforce_eager=False,
            pipeline_parallel_size=num_stages,
        )

        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_new_tokens,
        )
        outputs = llm.generate(prompt, sampling_params=sampling_params)
        if not outputs:
            raise RuntimeError("vLLM engine returned no outputs")
        output = outputs[0]
        text = output.outputs[0].text if output.outputs else ""
        stats = {
            "prompt_tokens": len(output.prompt_token_ids),
            "generated_tokens": len(output.outputs[0].token_ids) if output.outputs else 0,
            "finish_reason": output.outputs[0].finish_reason if output.outputs else None,
            "logprobs_available": bool(output.outputs and output.outputs[0].logprobs),
        }
        if log_stats:
            logger.info("Engine stats: %s", stats)
        return text, stats
    finally:
        shutil.rmtree(temp_model_dir, ignore_errors=True)
        # Restore original CUDA_VISIBLE_DEVICES if it was modified
        if gpu_ids is not None:
            if old_cuda_visible is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda_visible
            elif "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a quick vLLM inference test against exported pipeline stages."
    )
    parser.add_argument(
        "--pipeline-dir",
        type=str,
        default=DEFAULT_PIPELINE_DIR,
        help=f"Directory containing stage_* folders (default: {DEFAULT_PIPELINE_DIR})",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello from the vLLM offline pipeline test!",
        help="Prompt used for the sanity-check generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["float16", "bfloat16", "float32", "auto", "fp16", "bf16", "fp32"],
        help="Optional dtype override for vLLM engine.",
    )
    parser.add_argument(
        "--engine-log-stats",
        action="store_true",
        help="Print vLLM engine token/statistics summary after generation.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated list of GPU IDs to use for pipeline stages (e.g., '0,1,2,3'). "
        "If not specified, vLLM will auto-assign GPUs. Number of GPUs must match number of stages.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline_dir = Path(args.pipeline_dir).expanduser().resolve()
    if not pipeline_dir.exists():
        raise FileNotFoundError(f"Pipeline directory not found: {pipeline_dir}")

    stage_pairs = discover_stage_dirs(pipeline_dir)
    metadata = load_stage_metadata(stage_pairs)
    summarize_pipeline(metadata, pipeline_dir)
    validate_stage_layout(metadata)

    dtype = args.dtype if args.dtype not in (None, "auto") else None
    gpu_ids = None
    if args.gpus:
        gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpus.split(",")]
        if len(gpu_ids) != len(stage_pairs):
            raise ValueError(
                f"Number of GPU IDs ({len(gpu_ids)}) must match number of stages ({len(stage_pairs)})"
            )
        logger.info(f"Using GPUs: {gpu_ids} for {len(stage_pairs)} pipeline stages")
    
    decoded, stats = run_with_vllm_engine(
        pipeline_dir=pipeline_dir,
        num_stages=len(stage_pairs),
        metadata=metadata,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        dtype=dtype,
        log_stats=args.engine_log_stats,
        gpu_ids=gpu_ids,
    )

    if args.engine_log_stats:
        logger.info("Prompt tokens: %s, generated tokens: %s, finish_reason=%s", stats["prompt_tokens"], stats["generated_tokens"], stats["finish_reason"])

    print("\n=== vLLM Pipeline Test Output ===")
    print(decoded)
    print("================================")


if __name__ == "__main__":
    main()


