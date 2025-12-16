# Quick Start: Dynamic Loading from Full Model

This guide shows how to run vLLM pipeline parallelism from a complete model without pre-splitting.

## Prerequisites

- vLLM installed with ZeroMQ support
- A complete model (e.g., Llama-3-8B) in HuggingFace format
- Multiple GPUs available

## Step 1: Validate Layer Ranges

First, validate your layer range configuration:

```bash
# Check your model and auto-calculate ranges
python tools/offline_pipelining/validate_layer_ranges.py \
    --model /path/to/Llama-3-8B \
    --num-stages 4
```

**Expected output:**
```
✓ Model: /path/to/Llama-3-8B
✓ Detected 32 layers from config.json
✓ Auto-calculated layer ranges for 4 stages

Layer Range Configuration:
------------------------------------------------------------
  Stage 0: [ 0- 8)  ( 8 layers)
  Stage 1: [ 8-16)  ( 8 layers)
  Stage 2: [16-24)  ( 8 layers)
  Stage 3: [24-32)  ( 8 layers)
------------------------------------------------------------

✓ Configuration is VALID
  - All 32 layers are covered exactly once
  - No overlaps or gaps detected
  - Ready for pipeline parallelism
```

## Step 2: Launch Pipeline Stages

### Option A: Automated Script (Recommended)

1. Edit the configuration in `run_from_full_model.sh`:

```bash
nano tools/offline_pipelining/run_from_full_model.sh
```

2. Set your model path:

```bash
FULL_MODEL_PATH="/path/to/Llama-3-8B"  # Change this line
```

3. Run the script:

```bash
bash tools/offline_pipelining/run_from_full_model.sh
```

The script will automatically:
- Calculate layer ranges
- Launch all pipeline stages
- Configure ZeroMQ communication
- Start an HTTP API server on port 8000

### Option B: Manual Launch

Launch each stage in separate terminals:

**Terminal 1 - Stage 0:**
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/Llama-3-8B \
    --pipeline-stage-mode external \
    --pipeline-stage-idx 0 \
    --pipeline-total-stages 4 \
    --pipeline-layer-range "0-8" \
    --pipeline-local-bind-port 15550 \
    --pipeline-prev-stage-addr 127.0.0.1:15554 \
    --tensor-parallel-size 1 \
    --port 8000
```

**Terminal 2 - Stage 1:**
```bash
CUDA_VISIBLE_DEVICES=1 vllm serve /path/to/Llama-3-8B \
    --pipeline-stage-mode external \
    --pipeline-stage-idx 1 \
    --pipeline-total-stages 4 \
    --pipeline-layer-range "8-16" \
    --pipeline-local-bind-port 15550 \
    --pipeline-prev-stage-service-addr 127.0.0.1:15550 \
    --tensor-parallel-size 1 \
    --external-pp-worker
```

**Terminal 3 - Stage 2:**
```bash
CUDA_VISIBLE_DEVICES=2 vllm serve /path/to/Llama-3-8B \
    --pipeline-stage-mode external \
    --pipeline-stage-idx 2 \
    --pipeline-total-stages 4 \
    --pipeline-layer-range "16-24" \
    --pipeline-local-bind-port 15550 \
    --pipeline-prev-stage-service-addr 127.0.0.1:15550 \
    --tensor-parallel-size 1 \
    --external-pp-worker
```

**Terminal 4 - Stage 3:**
```bash
CUDA_VISIBLE_DEVICES=3 vllm serve /path/to/Llama-3-8B \
    --pipeline-stage-mode external \
    --pipeline-stage-idx 3 \
    --pipeline-total-stages 4 \
    --pipeline-layer-range "24-32" \
    --pipeline-local-bind-port 15554 \
    --pipeline-prev-stage-service-addr 127.0.0.1:15550 \
    --tensor-parallel-size 1 \
    --external-pp-worker
```

## Step 3: Test Inference

Once all stages are running, test with a simple request:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/Llama-3-8B",
    "prompt": "San Francisco is a",
    "max_tokens": 50,
    "temperature": 0
  }'
```

## Understanding the Output

### During Startup

You should see each stage loading only its assigned layers:

```
Stage 0: Loading layers 0-7... (2.1 GB)
Stage 1: Loading layers 8-15... (2.1 GB)  
Stage 2: Loading layers 16-23... (2.1 GB)
Stage 3: Loading layers 24-31... (2.1 GB)
```

**Key Points:**
- Each stage loads ~25% of the model (for 4 stages)
- Total memory = 4 × 2.1GB ≈ 8.4GB (same as loading full model once)
- Startup time: ~15-25 seconds per stage

### Memory Verification

Check GPU memory usage:

```bash
nvidia-smi
```

Expected:
- Each GPU should show ~2-3GB for a 8B model with 4 stages
- Not 8GB on each GPU (that would indicate full model loading)

## Troubleshooting

### "Invalid layer range" errors

Run the validator first:
```bash
python tools/offline_pipelining/validate_layer_ranges.py \
    --model /path/to/your-model \
    --ranges "0-8" "8-16" "16-24" "24-32"
```

### Timeout errors

Check if all stages are running:
```bash
ps aux | grep vllm
```

### Connection refused

Ensure stages are launched in order and have time to initialize (~30s).

### Check logs

If using the automated script:
```bash
tail -f zeromq_pp_full_model_logs/stage_*.log
```

## Comparing to Pre-Split Models

### Dynamic Loading (This Approach)

**Pros:**
- No pre-processing required
- Easy to change configuration
- No duplicate disk space

**Cons:**  
- 5-10s slower startup per stage
- All stages need model access

### Pre-Split Models

**Pros:**
- Faster startup
- Independent stage deployment

**Cons:**
- Requires `serve_pipeline.py` export step
- ~2× disk space required
- Must re-export to change config

**Recommendation:** Use dynamic loading for development and testing. Use pre-split for production deployments.

## Advanced Configuration

### Different Layer Distributions

For a 32-layer model with uneven distribution:

```bash
# Validate first
python tools/offline_pipelining/validate_layer_ranges.py \
    --total-layers 32 \
    --ranges "0-10" "10-18" "18-24" "24-32"

# Launch with custom ranges
vllm serve /path/to/model --pipeline-layer-range "0-10" ...
```

### Network Deployment

For stages on different machines, update ZeroMQ addresses:

```bash
# Stage 0 on machine A (192.168.1.10)
--pipeline-prev-stage-addr 192.168.1.13:15554  # Last stage on machine D

# Stage 1 on machine B (192.168.1.11)  
--pipeline-prev-stage-service-addr 192.168.1.10:15550  # Previous stage on machine A

# etc.
```

## Next Steps

- Read the full [README.md](README.md) for architecture details
- Try the HTTP entry point: `http_index.py`
- Explore hybrid tensor + pipeline parallelism configurations
