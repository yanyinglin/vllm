#!/bin/bash

# ============================================================================
# ZeroMQ Pipeline Parallelism - Run from Full Model (No Pre-Splitting)
# ============================================================================
# This script demonstrates how to run pipeline parallelism directly from a
# complete model without pre-splitting using serve_pipeline.py.
#
# Benefits:
# - No pre-processing step required
# - No additional disk space for split models
# - Easy configuration changes (just modify layer ranges)
#
# Trade-offs:
# - Slightly slower startup (~5-10s extra) due to scanning all weight files
# - All stages need access to the same full model directory
# ============================================================================

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Path to the FULL model (not split)
FULL_MODEL_PATH="/path/to/Llama-3-8B"  # CHANGE THIS

# Number of pipeline stages
NUM_STAGES=4

# Base port for ZeroMQ communication
BASE_PORT=15550

# GPU mapping (one GPU per stage)
GPU_IDS=(0 1 2 3)

# Local IP (auto-detect or set manually)
LOCAL_IP=${VLLM_HOST_IP:-$(hostname -I | awk '{print $1}')}

# API port for stage 0
API_PORT=8000

# vLLM source directory (if running from source)
VLLM_SOURCE_DIR=${VLLM_SOURCE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}

# Log directory
LOG_DIR="./zeromq_pp_full_model_logs"

# ============================================================================
# Layer Range Calculation
# ============================================================================

# Get total number of layers from model config
get_num_layers() {
    local model_path=$1
    local config_file="${model_path}/config.json"
    
    if [ -f "$config_file" ]; then
        # Try different config keys for different model architectures
        python3 -c "
import json
with open('$config_file') as f:
    config = json.load(f)
# Try common keys
for key in ['num_hidden_layers', 'n_layer', 'num_layers']:
    if key in config:
        print(config[key])
        exit(0)
print('32')  # Default fallback
"
    else
        echo "32"  # Default fallback
    fi
}

NUM_LAYERS=$(get_num_layers "$FULL_MODEL_PATH")
echo "Detected $NUM_LAYERS layers in model"

# Calculate layer ranges for each stage
calculate_layer_ranges() {
    local total_layers=$1
    local num_stages=$2
    local layer_ranges=()
    
    for ((stage=0; stage<num_stages; stage++)); do
        local start=$((total_layers * stage / num_stages))
        local end=$((total_layers * (stage + 1) / num_stages))
        layer_ranges+=("$start-$end")
    done
    
    echo "${layer_ranges[@]}"
}

LAYER_RANGES=($(calculate_layer_ranges $NUM_LAYERS $NUM_STAGES))

echo "Layer ranges for $NUM_STAGES stages:"
for ((i=0; i<NUM_STAGES; i++)); do
    echo "  Stage $i: ${LAYER_RANGES[$i]}"
done

# ============================================================================
# Validation
# ============================================================================

if [ ! -d "$FULL_MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $FULL_MODEL_PATH"
    echo "Please set FULL_MODEL_PATH to your complete model directory"
    exit 1
fi

if [ ! -f "$FULL_MODEL_PATH/config.json" ]; then
    echo "Error: config.json not found in $FULL_MODEL_PATH"
    exit 1
fi

# Check if model has safetensors or pytorch_model files
if ! ls "$FULL_MODEL_PATH"/*.safetensors >/dev/null 2>&1 && \
   ! ls "$FULL_MODEL_PATH"/pytorch_model*.bin >/dev/null 2>&1; then
    echo "Error: No model weight files found in $FULL_MODEL_PATH"
    echo "Expected .safetensors or pytorch_model*.bin files"
    exit 1
fi

# ============================================================================
# Cleanup
# ============================================================================

cleanup() {
    echo ""
    echo "Cleaning up..."
    for pid in "${STAGE_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping process $pid"
            kill "$pid" 2>/dev/null || true
        fi
    done
    wait
    echo "Cleanup complete"
}

trap cleanup EXIT INT TERM

# ============================================================================
# Setup
# ============================================================================

# Create log directory
mkdir -p "$LOG_DIR"
echo "Logs will be written to $LOG_DIR/"

# Array to track PIDs
STAGE_PIDS=()

# ============================================================================
# Launch Pipeline Stages
# ============================================================================

echo ""
echo "============================================================================"
echo "Launching $NUM_STAGES pipeline stages from full model"
echo "Model: $FULL_MODEL_PATH"
echo "Local IP: $LOCAL_IP"
echo "Base Port: $BASE_PORT"
echo "============================================================================"
echo ""

# Determine vLLM command
if [ -d "$VLLM_SOURCE_DIR/vllm" ]; then
    echo "Using vLLM from source: $VLLM_SOURCE_DIR"
    export PYTHONPATH="$VLLM_SOURCE_DIR:$PYTHONPATH"
    VLLM_CMD="python -m vllm.entrypoints.cli.main"
else
    echo "Using installed vLLM"
    VLLM_CMD="vllm"
fi

# Launch each stage
for ((stage=0; stage<NUM_STAGES; stage++)); do
    gpu_id=${GPU_IDS[$stage]}
    layer_range=${LAYER_RANGES[$stage]}
    log_file="$LOG_DIR/stage_${stage}.log"
    
    echo "Starting Stage $stage (GPU $gpu_id, layers $layer_range)..."
    
    # Build command
    cmd="CUDA_VISIBLE_DEVICES=$gpu_id $VLLM_CMD serve"
    cmd="$cmd $FULL_MODEL_PATH"
    cmd="$cmd --pipeline-stage-mode external"
    cmd="$cmd --pipeline-stage-idx $stage"
    cmd="$cmd --pipeline-total-stages $NUM_STAGES"
    cmd="$cmd --pipeline-layer-range $layer_range"
    cmd="$cmd --tensor-parallel-size 1"
    
    # Stage-specific configuration
    if [ $stage -eq 0 ]; then
        # Stage 0: Has API server + connects to last stage for return path
        last_stage_return_port=$((BASE_PORT + NUM_STAGES))
        cmd="$cmd --pipeline-local-bind-port $BASE_PORT"
        cmd="$cmd --pipeline-prev-stage-addr ${LOCAL_IP}:${last_stage_return_port}"
        cmd="$cmd --port $API_PORT"
        echo "  → API server on port $API_PORT"
        echo "  → PUSH socket binding on port $BASE_PORT"
        echo "  → Return PULL from ${LOCAL_IP}:${last_stage_return_port}"
    elif [ $stage -eq $((NUM_STAGES - 1)) ]; then
        # Last stage: Connects to previous + binds return socket
        last_stage_return_port=$((BASE_PORT + NUM_STAGES))
        cmd="$cmd --pipeline-local-bind-port $last_stage_return_port"
        cmd="$cmd --pipeline-prev-stage-service-addr ${LOCAL_IP}:${BASE_PORT}"
        cmd="$cmd --external-pp-worker"
        echo "  → PULL from ${LOCAL_IP}:${BASE_PORT}"
        echo "  → Return PUSH binding on port $last_stage_return_port"
    else
        # Middle stages: Connect to previous + bind for next
        cmd="$cmd --pipeline-local-bind-port $BASE_PORT"
        cmd="$cmd --pipeline-prev-stage-service-addr ${LOCAL_IP}:${BASE_PORT}"
        cmd="$cmd --external-pp-worker"
        echo "  → PULL from ${LOCAL_IP}:${BASE_PORT}"
        echo "  → PUSH socket binding on port $BASE_PORT"
    fi
    
    # Launch in background
    bash -c "$cmd" > "$log_file" 2>&1 &
    pid=$!
    STAGE_PIDS+=($pid)
    
    echo "  → PID: $pid, Log: $log_file"
    echo ""
    
    # Wait between stages to ensure orderly startup
    sleep 2
done

# ============================================================================
# Wait for Initialization
# ============================================================================

echo "Waiting for all stages to initialize..."
sleep 30

# ============================================================================
# Health Check
# ============================================================================

echo ""
echo "Checking stage 0 API server..."
if curl -s "http://localhost:$API_PORT/health" >/dev/null 2>&1; then
    echo "✓ Stage 0 API server is ready"
else
    echo "✗ Stage 0 API server not responding"
    echo "Check logs in $LOG_DIR/stage_0.log"
fi

# ============================================================================
# Test Inference (Optional)
# ============================================================================

echo ""
echo "============================================================================"
echo "Pipeline is running. Test with:"
echo "============================================================================"
echo ""
echo "curl http://localhost:$API_PORT/v1/completions \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{"
echo "    \"model\": \"$FULL_MODEL_PATH\","
echo "    \"prompt\": \"San Francisco is a\","
echo "    \"max_tokens\": 50,"
echo "    \"temperature\": 0"
echo "  }'"
echo ""
echo "============================================================================"
echo "Process Information:"
echo "============================================================================"
for ((i=0; i<NUM_STAGES; i++)); do
    echo "Stage $i: PID ${STAGE_PIDS[$i]}, GPU ${GPU_IDS[$i]}, Layers ${LAYER_RANGES[$i]}"
done
echo ""
echo "Logs: $LOG_DIR/"
echo "Press Ctrl+C to stop all stages"
echo "============================================================================"

# Keep script running
wait
