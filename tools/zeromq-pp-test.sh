#!/bin/bash

# ZeroMQ Pipeline Parallelism Test Script
# This script launches 4 vLLM stages in external mode using ZeroMQ for communication

set -euo pipefail

# Configuration
BASE_MODEL_PATH="/home/yanying/pipeline_export/Llama-3-8B"
NUM_STAGES=4
BASE_PORT=15550  # Starting port for ZeroMQ communication

# GPU mapping: one GPU per stage.
# Default: stage 0→GPU0, stage 1→GPU1, ...
# Adjust as needed, but must provide at least NUM_STAGES entries.
GPU_IDS=(0 1 2 3)

# Source code mode: if VLLM_SOURCE_DIR is set, use source code directly
# Example: export VLLM_SOURCE_DIR="/home/yanying/workspace/github/vllm"
export VLLM_SOURCE_DIR="/home/yanying/workspace/github/vllm"
# If not set, will use installed vllm command
VLLM_SOURCE_DIR="${VLLM_SOURCE_DIR:-}"
vllm --
# Get local IP address
get_local_ip() {
    # Try to get IP from environment variable first
    if [ -n "${VLLM_HOST_IP:-}" ]; then
        echo "$VLLM_HOST_IP"
        return
    fi
    
    # Try to get IP from network interface
    if command -v ip >/dev/null 2>&1; then
        # Get the IP of the default route interface
        local ip=$(ip route get 8.8.8.8 2>/dev/null | grep -oP 'src \K\S+' | head -1)
        if [ -n "$ip" ]; then
            echo "$ip"
            return
        fi
    fi
    
    # Fallback to hostname -I
    if command -v hostname >/dev/null 2>&1; then
        local ip=$(hostname -I | awk '{print $1}')
        if [ -n "$ip" ]; then
            echo "$ip"
            return
        fi
    fi
    
    # Final fallback
    echo "127.0.0.1"
}

LOCAL_IP=$(get_local_ip)

# Determine vLLM command based on source mode
if [ -n "$VLLM_SOURCE_DIR" ]; then
    if [ ! -d "$VLLM_SOURCE_DIR" ]; then
        echo "Error: VLLM_SOURCE_DIR does not exist: $VLLM_SOURCE_DIR"
        exit 1
    fi
    VLLM_CMD="python -m vllm.entrypoints.cli.main serve"
    export PYTHONPATH="${VLLM_SOURCE_DIR}:${PYTHONPATH:-}"
    echo "=========================================="
    echo "ZeroMQ Pipeline Parallelism Test"
    echo "=========================================="
    echo "Mode: Using source code"
    echo "Source Directory: $VLLM_SOURCE_DIR"
    echo "Local IP: $LOCAL_IP"
    echo "Number of Stages: $NUM_STAGES"
    echo "Base Model Path: $BASE_MODEL_PATH"
    echo "Base ZeroMQ Port: $BASE_PORT"
    echo "=========================================="
    echo ""
else
    VLLM_CMD="vllm serve"
    echo "=========================================="
    echo "ZeroMQ Pipeline Parallelism Test"
    echo "=========================================="
    echo "Mode: Using installed vllm"
    echo "Local IP: $LOCAL_IP"
    echo "Number of Stages: $NUM_STAGES"
    echo "Base Model Path: $BASE_MODEL_PATH"
    echo "Base ZeroMQ Port: $BASE_PORT"
    echo "=========================================="
    echo ""
    echo "Tip: To use source code, set VLLM_SOURCE_DIR environment variable"
    echo "     Example: export VLLM_SOURCE_DIR=\"/home/yanying/workspace/github/vllm\""
    echo ""
fi

# Check if model directories exist
for i in $(seq 0 $((NUM_STAGES - 1))); do
    STAGE_DIR="${BASE_MODEL_PATH}/stage_${i}"
    if [ ! -d "$STAGE_DIR" ]; then
        echo "Error: Stage directory not found: $STAGE_DIR"
        exit 1
    fi
done

# Validate GPU mapping
if [ "${#GPU_IDS[@]}" -lt "$NUM_STAGES" ]; then
    echo "Error: GPU_IDS has only ${#GPU_IDS[@]} entries, but NUM_STAGES=$NUM_STAGES"
    echo "Please set GPU_IDS to have at least one GPU id per stage."
    exit 1
fi

# Create log directory
LOG_DIR="./zeromq_pp_logs"
mkdir -p "$LOG_DIR"

# Array to store PIDs of launched processes
PIDS=()

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Killing process $pid"
            kill "$pid" 2>/dev/null || true
        fi
    done
    wait
    echo "Cleanup complete"
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

# Launch each stage
for stage_idx in $(seq 0 $((NUM_STAGES - 1))); do
    STAGE_DIR="${BASE_MODEL_PATH}/stage_${stage_idx}"
    LOG_FILE="${LOG_DIR}/stage_${stage_idx}.log"
    STAGE_GPU="${GPU_IDS[$stage_idx]}"
    
    # Determine if this is first or last stage
    IS_FIRST=$([ "$stage_idx" -eq 0 ] && echo "true" || echo "false")
    IS_LAST=$([ "$stage_idx" -eq $((NUM_STAGES - 1)) ] && echo "true" || echo "false")
    
    # Build vLLM command
    CMD="$VLLM_CMD"
    CMD="$CMD $STAGE_DIR"
    CMD="$CMD --pipeline-stage-mode external"
    CMD="$CMD --pipeline-stage-idx $stage_idx"
    CMD="$CMD --pipeline-total-stages $NUM_STAGES"
    
    # Bind mode configuration:
    # - PUSH sockets bind to ports (senders)
    # - PULL sockets connect to Service addresses (receivers)
    # - This allows Kubernetes Service load balancing
    
    # Forward path: stage_i binds PUSH, stage_{i+1} connects PULL
    if [ "$IS_LAST" = "false" ]; then
        # Non-last stages: bind PUSH socket to local port
        BIND_PORT=$((BASE_PORT + stage_idx))
        CMD="$CMD --pipeline-local-bind-port $BIND_PORT"
    fi
    
    if [ "$IS_FIRST" = "false" ]; then
        # Non-first stages: connect PULL socket to previous stage's PUSH Service
        PREV_BIND_PORT=$((BASE_PORT + stage_idx - 1))
        CMD="$CMD --pipeline-next-stage-addr ${LOCAL_IP}:${PREV_BIND_PORT}"
    fi
    
    # Return path: last stage binds PUSH, stage 0 connects PULL
    if [ "$IS_LAST" = "true" ]; then
        # Last stage: bind PUSH socket for return path
        RETURN_BIND_PORT=$((BASE_PORT + NUM_STAGES))
        CMD="$CMD --pipeline-local-bind-port $RETURN_BIND_PORT"
    fi
    
    if [ "$IS_FIRST" = "true" ] && [ "$IS_LAST" = "false" ]; then
        # Stage 0: connect PULL socket to last stage's return Service
        RETURN_BIND_PORT=$((BASE_PORT + NUM_STAGES))
        CMD="$CMD --pipeline-prev-stage-addr ${LOCAL_IP}:${RETURN_BIND_PORT}"
    fi
    
    # Add tensor parallel size if needed (default to 1)
    CMD="$CMD --tensor-parallel-size 1"
    
    # Disable CUDA graphs to debug pipeline parallelism issues
    # CMD="$CMD --enforce-eager"
    
    # Layer range is now automatically detected from _pipeline_info in config.json
    # If _pipeline_info exists, vLLM will use all local layers [0, num_hidden_layers)
    # No need to manually specify --pipeline-layer-range anymore
    # CMD="$CMD --pipeline-layer-range 0-8"
    
    # Stage 0 exposes HTTP API; later stages run as external PP workers.
    if [ "$IS_FIRST" = "true" ]; then
        API_PORT=$((8000 + stage_idx))
        CMD="$CMD --port $API_PORT"
    else
        CMD="$CMD --external-pp-worker"
    fi
    
    echo "Launching Stage $stage_idx:"
    echo "  Model: $STAGE_DIR"
    echo "  Log: $LOG_FILE"
    echo "  GPU: $STAGE_GPU"
    if [ "$IS_LAST" = "false" ]; then
        BIND_PORT=$((BASE_PORT + stage_idx))
        echo "  Forward Bind Port: $BIND_PORT (PUSH socket binds, allows next stage to connect)"
    fi
    if [ "$IS_FIRST" = "false" ]; then
        PREV_BIND_PORT=$((BASE_PORT + stage_idx - 1))
        echo "  Previous Stage Service: ${LOCAL_IP}:${PREV_BIND_PORT} (PULL socket connects)"
    fi
    if [ "$IS_LAST" = "true" ]; then
        RETURN_BIND_PORT=$((BASE_PORT + NUM_STAGES))
        echo "  Return Bind Port: $RETURN_BIND_PORT (PUSH socket binds for return path)"
    fi
    if [ "$IS_FIRST" = "true" ] && [ "$IS_LAST" = "false" ]; then
        RETURN_BIND_PORT=$((BASE_PORT + NUM_STAGES))
        echo "  Return Service: ${LOCAL_IP}:${RETURN_BIND_PORT} (PULL socket connects to last stage)"
    fi
    if [ "$IS_FIRST" = "true" ]; then
        echo "  API Port: $API_PORT"
    else
        echo "  Mode: external-pp-worker (no HTTP API)"
    fi
    echo "  Command: $CMD"
    echo ""
    
    # Launch stage in background, binding this stage to its GPU
    CUDA_VISIBLE_DEVICES="$STAGE_GPU" eval "$CMD" > "$LOG_FILE" 2>&1 &
    PID=$!
    PIDS+=($PID)
    
    echo "Stage $stage_idx launched with PID: $PID"
    
    # Wait a bit and check if process is still running
    sleep 3
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "ERROR: Stage $stage_idx (PID: $PID) failed to start!"
        echo "Check log file: $LOG_FILE"
        tail -20 "$LOG_FILE"
        exit 1
    fi
    
    echo "Stage $stage_idx is running"
    echo ""
done

echo ""
echo "=========================================="
echo "All stages launched successfully!"
echo "=========================================="
echo "Stage PIDs: ${PIDS[*]}"
echo "Logs directory: $LOG_DIR"
echo ""
echo "API endpoint (for testing):"
echo "  Stage 0: http://localhost:8000"
echo ""
echo "ZeroMQ communication ports (bind mode):"
echo "Forward path (stage_k binds PUSH, stage_{k+1} connects PULL):"
for stage_idx in $(seq 0 $((NUM_STAGES - 2))); do
    BIND_PORT=$((BASE_PORT + stage_idx))
    echo "  Stage $stage_idx binds PUSH on port $BIND_PORT"
    echo "  Stage $((stage_idx + 1)) connects PULL to ${LOCAL_IP}:${BIND_PORT}"
done
RETURN_BIND_PORT=$((BASE_PORT + NUM_STAGES))
echo "Return path (last stage binds PUSH, stage 0 connects PULL):"
echo "  Stage $((NUM_STAGES - 1)) binds PUSH on port $RETURN_BIND_PORT"
echo "  Stage 0 connects PULL to ${LOCAL_IP}:${RETURN_BIND_PORT}"
echo ""
echo "To view logs:"
echo "  tail -f $LOG_DIR/stage_0.log"
echo "  tail -f $LOG_DIR/stage_1.log"
echo "  ..."
echo ""


echo "Waiting 60 seconds for all stages to fully initialize..."
sleep 30
echo ""

# Run a simple test inference on the stage 0 HTTP API
TEST_API_PORT=8000
echo "Running a test inference against stage 0 on port ${TEST_API_PORT} ..."

# Wait for API to be ready with health check
echo "Waiting for API server to be ready..."
MAX_RETRIES=30
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    set +e
    HEALTH_RESPONSE=$(curl -sS -X GET "http://localhost:${TEST_API_PORT}/health" 2>&1)
    HEALTH_STATUS=$?
    set -e
    
    if [ "$HEALTH_STATUS" -eq 0 ]; then
        echo "API server is ready"
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
        echo "  Waiting for API server... (attempt $RETRY_COUNT/$MAX_RETRIES)"
        sleep 2
    fi
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "Warning: API server health check failed after $MAX_RETRIES attempts"
    echo "Continuing anyway..."
fi

# First, get the available model name from the API
echo "Querying available models..."
set +e
MODELS_RESPONSE=$(curl -sS -X GET "http://localhost:${TEST_API_PORT}/v1/models" 2>&1)
MODELS_STATUS=$?
set -e

if [ "$MODELS_STATUS" -ne 0 ]; then
    echo "Warning: Failed to query /v1/models (curl exit code: $MODELS_STATUS)"
    echo "Response: $MODELS_RESPONSE"
    echo "Using model path as fallback..."
    MODEL_NAME="${BASE_MODEL_PATH}/stage_0"
else
    # Extract model ID from response (assuming JSON format)
    MODEL_NAME=$(echo "$MODELS_RESPONSE" | grep -oP '"id"\s*:\s*"[^"]*"' | head -1 | cut -d'"' -f4 || echo "")
    if [ -z "$MODEL_NAME" ]; then
        echo "Warning: Could not parse model name from /v1/models response"
        echo "Response: $MODELS_RESPONSE"
        echo "Using model path as fallback..."
        MODEL_NAME="${BASE_MODEL_PATH}/stage_0"
    else
        echo "Found model: $MODEL_NAME"
    fi
fi

# Use the model name in the test request
TEST_PAYLOAD=$(cat <<EOF
{
  "model": "${MODEL_NAME}",
  "prompt": "Hello from ZeroMQ pipeline test",
  "max_tokens": 64
}
EOF
)

echo "Sending test completion request..."
set +e
TEST_RESPONSE=$(curl -sS -X POST "http://localhost:${TEST_API_PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "${TEST_PAYLOAD}" 2>&1)
TEST_STATUS=$?
set -e

if [ "$TEST_STATUS" -ne 0 ]; then
    echo "Test inference request failed (curl exit code: $TEST_STATUS)"
    echo "Response:"
    echo "$TEST_RESPONSE"
else
    # Check if response contains an error
    if echo "$TEST_RESPONSE" | grep -q '"error"'; then
        echo "Test inference returned an error:"
        echo "$TEST_RESPONSE"
    else
        echo "Test inference response:"
        echo "$TEST_RESPONSE"
    fi
fi
echo ""

echo "Press Ctrl+C to stop all stages"
echo "=========================================="
echo ""

# Wait for all processes
wait

