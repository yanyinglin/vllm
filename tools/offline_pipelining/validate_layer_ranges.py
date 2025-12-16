#!/usr/bin/env python3
"""
Validate Pipeline Layer Range Configuration

This script validates that pipeline layer ranges are correctly configured:
- No overlapping layers between stages
- All layers are covered exactly once
- Layer ranges are contiguous and valid

Usage:
    python validate_layer_ranges.py --total-layers 32 --ranges "0-8" "8-16" "16-24" "24-32"
    python validate_layer_ranges.py --model /path/to/model --num-stages 4
"""

import argparse
import json
import sys
from pathlib import Path


def parse_layer_range(range_str: str) -> tuple[int, int]:
    """Parse layer range string like '0-8' or '[0-8]' to (start, end) tuple."""
    range_str = range_str.strip()
    # Remove optional brackets
    if range_str.startswith('[') and range_str.endswith(']'):
        range_str = range_str[1:-1]
    
    parts = range_str.split('-')
    if len(parts) != 2:
        raise ValueError(f"Invalid layer range format: {range_str}. Expected 'start-end'")
    
    try:
        start = int(parts[0])
        end = int(parts[1])
    except ValueError:
        raise ValueError(f"Invalid layer range format: {range_str}. Start and end must be integers")
    
    if start < 0 or end < 0:
        raise ValueError(f"Invalid layer range: {range_str}. Layer indices must be non-negative")
    
    if start >= end:
        raise ValueError(f"Invalid layer range: {range_str}. Start must be less than end")
    
    return start, end


def get_num_layers_from_model(model_path: str) -> int:
    """Get number of hidden layers from model config.json."""
    config_path = Path(model_path) / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Try different config keys for different model architectures
    for key in ['num_hidden_layers', 'n_layer', 'num_layers', 'n_layers']:
        if key in config:
            return config[key]
    
    raise ValueError(f"Could not find layer count in config: {config_path}")


def auto_calculate_ranges(total_layers: int, num_stages: int) -> list[tuple[int, int]]:
    """Automatically calculate layer ranges for given number of stages."""
    ranges = []
    for stage in range(num_stages):
        start = total_layers * stage // num_stages
        end = total_layers * (stage + 1) // num_stages
        ranges.append((start, end))
    return ranges


def validate_ranges(ranges: list[tuple[int, int]], total_layers: int) -> tuple[bool, list[str]]:
    """
    Validate that layer ranges are correct.
    
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    if not ranges:
        errors.append("No layer ranges provided")
        return False, errors
    
    # Check each range is valid
    for i, (start, end) in enumerate(ranges):
        if start >= end:
            errors.append(f"Stage {i}: Invalid range [{start}-{end}), start >= end")
        if start < 0:
            errors.append(f"Stage {i}: Invalid range [{start}-{end}), negative start")
        if end > total_layers:
            errors.append(f"Stage {i}: Invalid range [{start}-{end}), end > total_layers ({total_layers})")
    
    # Sort ranges by start index
    sorted_ranges = sorted(enumerate(ranges), key=lambda x: x[1][0])
    
    # Check for gaps and overlaps
    expected_next = 0
    for stage_idx, (start, end) in sorted_ranges:
        if start < expected_next:
            errors.append(
                f"Stage {stage_idx}: Overlapping range [{start}-{end}), "
                f"expected to start at {expected_next} or later"
            )
        elif start > expected_next:
            errors.append(
                f"Gap detected: Layers [{expected_next}-{start}) not assigned to any stage"
            )
        expected_next = max(expected_next, end)
    
    # Check if all layers are covered
    if expected_next < total_layers:
        errors.append(
            f"Incomplete coverage: Layers [{expected_next}-{total_layers}) not assigned to any stage"
        )
    
    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(
        description="Validate pipeline layer range configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate explicit ranges
  %(prog)s --total-layers 32 --ranges "0-8" "8-16" "16-24" "24-32"
  
  # Auto-calculate and validate from model
  %(prog)s --model /path/to/Llama-3-8B --num-stages 4
  
  # Validate custom ranges against model
  %(prog)s --model /path/to/Llama-3-8B --ranges "0-10" "10-20" "20-32"
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--total-layers',
        type=int,
        help='Total number of layers in the model'
    )
    input_group.add_argument(
        '--model',
        type=str,
        help='Path to model directory (will read config.json)'
    )
    
    # Range options
    parser.add_argument(
        '--ranges',
        nargs='+',
        help='Layer ranges for each stage (e.g., "0-8" "8-16" "16-24" "24-32")'
    )
    parser.add_argument(
        '--num-stages',
        type=int,
        help='Number of pipeline stages (auto-calculate ranges)'
    )
    
    args = parser.parse_args()
    
    # Determine total layers
    if args.model:
        try:
            total_layers = get_num_layers_from_model(args.model)
            print(f"✓ Model: {args.model}")
            print(f"✓ Detected {total_layers} layers from config.json")
        except Exception as e:
            print(f"✗ Error reading model config: {e}", file=sys.stderr)
            return 1
    else:
        total_layers = args.total_layers
        print(f"✓ Total layers: {total_layers}")
    
    # Determine ranges
    if args.ranges:
        # Parse explicit ranges
        try:
            ranges = [parse_layer_range(r) for r in args.ranges]
            print(f"✓ Validating {len(ranges)} explicit layer ranges")
        except ValueError as e:
            print(f"✗ Error parsing ranges: {e}", file=sys.stderr)
            return 1
    elif args.num_stages:
        # Auto-calculate ranges
        ranges = auto_calculate_ranges(total_layers, args.num_stages)
        print(f"✓ Auto-calculated layer ranges for {args.num_stages} stages")
    else:
        print("✗ Error: Must provide either --ranges or --num-stages", file=sys.stderr)
        return 1
    
    # Display ranges
    print("\nLayer Range Configuration:")
    print("-" * 60)
    for i, (start, end) in enumerate(ranges):
        num_layers = end - start
        print(f"  Stage {i}: [{start:2d}-{end:2d})  ({num_layers:2d} layers)")
    print("-" * 60)
    
    # Validate
    is_valid, errors = validate_ranges(ranges, total_layers)
    
    if is_valid:
        print("\n✓ Configuration is VALID")
        print(f"  - All {total_layers} layers are covered exactly once")
        print(f"  - No overlaps or gaps detected")
        print(f"  - Ready for pipeline parallelism")
        return 0
    else:
        print("\n✗ Configuration is INVALID")
        print("\nErrors found:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease fix the layer range configuration and try again.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
