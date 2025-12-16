# ZeroMQ Pipeline Parallelism Migration

## Overview

Starting from commit `cbfc1eb`, we migrated pipeline parallelism data transmission from NCCL to ZeroMQ. This change enables more flexible pipeline stage communication, especially for offline pipeline inference scenarios.

## Changes Summary

### Implementation Approach

**External Pipeline Mode** (Core code modifications)
- Added in commit `83e77c4cb`
- Integrated into vLLM core with new configuration options
- Requires modifications to core vLLM files
- Uses Bind Mode architecture: PUSH sockets bind to ports, PULL sockets connect to addresses
- More robust and maintainable long-term

### Key Files Added/Modified

#### Tools Directory

1. **`tools/offline_pipelining/zmq_communicator.py`**
   - Implements `ZeroMQCommunicator` class
   - Provides `send_tensor_dict()` and `recv_tensor_dict()` interfaces compatible with PyTorch distributed API
   - Uses ZeroMQ PUSH/PULL sockets for unidirectional communication between stages
   - Supports bidirectional communication for autoregressive generation (token ID feedback)

2. **`tools/offline_pipelining/test_pipeline.py`**
   - Implements `PipelineStage` and `ZeroMQPipeline` classes
   - Uses `ZeroMQCommunicator` for inter-stage communication
   - Supports multiprocess pipeline execution

3. **`tools/offline_pipelining/serve_pipeline.py`**
   - Pipeline export tool for splitting models into stages
   - Exports each stage as HuggingFace format

4. **`tools/offline_pipelining/test_model_zeromq.py`**
   - Test tool for validating pipeline stages with vLLM engine

5. **`tools/offline_pipelining/http_index.py`**
   - HTTP entry point for ZeroMQ pipeline parallelism
   - Launches all pipeline stages using external pipeline mode
   - Exposes HTTP server that forwards inference requests to stage 0
   - Handles process lifecycle and cleanup
   - Provides health check and pipeline status endpoints

6. **`tools/offline_pipelining/run_from_full_model.sh`** ⭐ NEW
   - Automated script for running pipeline parallelism from complete model (no pre-splitting)
   - Auto-calculates layer ranges for each stage
   - Launches all stages with correct ZeroMQ configuration
   - Demonstrates dynamic loading with `--pipeline-layer-range` parameter

7. **`tools/offline_pipelining/validate_layer_ranges.py`** ⭐ NEW
   - Validation tool for pipeline layer range configuration
   - Checks for overlaps, gaps, and invalid ranges
   - Supports auto-calculation from model config
   - Helps prevent configuration errors before deployment

#### Core vLLM Files (Modified in commit `83e77c4cb`)

**Note**: These modifications are required for the "External Pipeline Mode" feature.


1. **`vllm/v1/worker/gpu_model_runner.py`** (MODIFIED)
   - Added `execute_external_pipeline_step()` method for external PP stages
   - Added logic to handle external pipeline mode in `execute_model()`
   - Added CPU tensor transfer for ZeroMQ serialization (comments mention ZeroMQ)
   - Added error handling for ZeroMQ deserialization issues
   - **Lines modified**: ~1000 lines added/modified

2. **`vllm/distributed/device_communicators/zeromq_communicator.py`** (NEW)
   - Official ZeroMQ communicator for external pipeline mode
   - Similar interface to `tools/offline_pipelining/zmq_communicator.py` but integrated into vLLM core

3. **`vllm/config/parallel.py`** (MODIFIED)
   - Added `pipeline_stage_mode` configuration option ("internal" vs "external")
   - Added external pipeline mode configuration options:
     - `pipeline_stage_idx`: Current stage index
     - `pipeline_total_stages`: Total number of stages
     - `pipeline_layer_range`: Layer range for current stage
     - `pipeline_prev_stage_service_addr`: Previous stage's PUSH Service address (IP:port) for PULL socket to connect (bind mode)
     - `pipeline_next_stage_addr`: [DEPRECATED] Backward compatibility alias for `pipeline_prev_stage_service_addr`
     - `pipeline_prev_stage_addr`: Previous stage address
     - `pipeline_local_listen_port`: Local listen port
     - `pipeline_local_bind_port`: Local bind port for PUSH sockets (bind mode)

4. **`vllm/distributed/parallel_state.py`** (MODIFIED)
   - Added support for external pipeline mode in parallel state management

5. **`vllm/model_executor/models/utils.py`** (MODIFIED)
   - Added `PPMissingLayer` and related utilities for external pipeline mode

6. **`vllm/model_executor/models/llama.py`** (MODIFIED)
   - Added support for external pipeline mode layer ranges

7. **`vllm/model_executor/models/transformers/base.py`** (MODIFIED)
   - Added support for external pipeline mode in base transformer model

8. **`vllm/entrypoints/cli/serve.py`** (MODIFIED)
   - Added CLI arguments for external pipeline mode configuration

## Architecture

### Communication Flow

```
Stage 0 (First)          Stage 1 (Middle)          Stage N (Last)
   |                          |                         |
   |--[PUSH socket]---------->|--[PUSH socket]---------->|
   |                          |                         |
   |<--[PULL socket]----------|<--[PULL socket]---------|
   |                          |                         |
   |  (backward for tokens)    |                         |
```

### ZeroMQ Socket Pattern (Bind Mode)

- **Forward direction**: PUSH/PULL sockets
  - Stage i (non-last) binds PUSH socket to port `BASE_PORT + i`
  - Stage i+1 connects PULL socket to `Stage_i_IP:BASE_PORT+i`
  
- **Return direction**: For final results
  - Last stage binds PUSH socket to port `BASE_PORT + NUM_STAGES`
  - Stage 0 connects PULL socket to `Last_Stage_IP:BASE_PORT+NUM_STAGES`

### Data Serialization

- Tensor metadata (shape, dtype, device) serialized with pickle
- Tensor data serialized with `torch.save()` (CPU transfer)
- Multi-part messages: `[metadata_size, metadata, num_tensors, tensor1_size, tensor1, ...]`

## Benefits of ZeroMQ over NCCL

1. **No GPU Memory Overhead**: NCCL requires GPU memory buffers for communication (typically 50-100MB per group)
2. **Simpler Setup**: No need for NCCL initialization, unique IDs, or process group coordination
3. **Flexible Topology**: Easy to add/remove stages without full system restart
4. **Cross-Device Support**: Can work with CPU-only stages or mixed CPU/GPU setups
5. **Network Flexibility**: Can work over network (TCP) or local (IPC) connections

## Implementation Details

### External Pipeline Mode

**Added in**: commit `83e77c4cb`

This approach integrates ZeroMQ support directly into vLLM core:

1. Added `pipeline_stage_mode` configuration option
2. Modified `gpu_model_runner.py` to handle external pipeline stages
3. Created `vllm/distributed/device_communicators/zeromq_communicator.py`
4. Added CLI arguments and configuration validation
5. Uses Bind Mode architecture for flexible communication

**Key Features**:
- ✅ Fully integrated with vLLM configuration system
- ✅ More robust and maintainable long-term
- ✅ Better error handling and validation
- ✅ Official support path
- ✅ Bind Mode: PUSH sockets bind to ports, PULL sockets connect to addresses
- ✅ Supports Kubernetes Service load balancing

## Usage

vLLM supports two approaches for pipeline parallelism with ZeroMQ:

1. **Pre-Split Models** (Recommended for production): Export model stages once, then use them multiple times
2. **Dynamic Loading from Full Model** (Recommended for development): Run directly from complete model without pre-splitting

### Approach 1: Pre-Split Models (Traditional)

#### Step 1: Export Pipeline Stages

```bash
python tools/offline_pipelining/serve_pipeline.py \
    --model /path/to/model \
    --pipeline-parallel-size 4 \
    --output-dir /nfs_ssd/yanying/models/pipeline/Meta-Llama-3-8B/inference
```

**Pros**:
- ✅ Faster startup time (5-10s faster per stage)
- ✅ Minimal disk I/O during initialization
- ✅ Ideal for production deployments with fixed configurations

**Cons**:
- ❌ Requires pre-processing step
- ❌ Additional disk space (duplicate of original model)
- ❌ Need to re-export if changing number of stages

#### Step 2: Run Pre-Split Pipeline Stages

See sections below for running options.

### Approach 2: Dynamic Loading from Full Model (New)

**No pre-splitting required!** Each stage loads only its assigned layers from the complete model.

#### Quick Start

```bash
# Edit the script to set your model path
nano tools/offline_pipelining/run_from_full_model.sh

# Set FULL_MODEL_PATH to your complete model
FULL_MODEL_PATH="/path/to/Llama-3-8B"

# Run the script
bash tools/offline_pipelining/run_from_full_model.sh
```

The script will:
- ✅ Automatically calculate layer ranges for each stage
- ✅ Launch all pipeline stages with correct ZeroMQ configuration
- ✅ Each stage loads only its assigned layers (memory-efficient)
- ✅ Provide health checks and test commands

**Pros**:
- ✅ No pre-processing required
- ✅ No additional disk space needed
- ✅ Easy to experiment with different pipeline configurations
- ✅ Memory-efficient: Only loads needed layers per stage (using lazy loading)

**Cons**:
- ⚠️ Slightly slower startup (~5-10s extra per stage) due to scanning weight files
- ⚠️ All stages need access to the same model directory (shared storage or NFS)

#### Manual Launch Example

```bash
# Stage 0 (layers 0-8, first stage with API)
CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/Llama-3-8B \
    --pipeline-stage-mode external \
    --pipeline-stage-idx 0 \
    --pipeline-total-stages 4 \
    --pipeline-layer-range "0-8" \
    --pipeline-local-bind-port 15550 \
    --pipeline-prev-stage-addr 127.0.0.1:15554 \
    --tensor-parallel-size 1 \
    --port 8000

# Stage 1 (layers 8-16, middle stage)
CUDA_VISIBLE_DEVICES=1 vllm serve /path/to/Llama-3-8B \
    --pipeline-stage-mode external \
    --pipeline-stage-idx 1 \
    --pipeline-total-stages 4 \
    --pipeline-layer-range "8-16" \
    --pipeline-local-bind-port 15550 \
    --pipeline-prev-stage-service-addr 127.0.0.1:15550 \
    --tensor-parallel-size 1 \
    --external-pp-worker

# Stage 2 (layers 16-24, middle stage)
CUDA_VISIBLE_DEVICES=2 vllm serve /path/to/Llama-3-8B \
    --pipeline-stage-mode external \
    --pipeline-stage-idx 2 \
    --pipeline-total-stages 4 \
    --pipeline-layer-range "16-24" \
    --pipeline-local-bind-port 15550 \
    --pipeline-prev-stage-service-addr 127.0.0.1:15550 \
    --tensor-parallel-size 1 \
    --external-pp-worker

# Stage 3 (layers 24-32, last stage)
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

#### How Dynamic Loading Works

**Memory-Efficient Lazy Loading**:
1. ✅ Opens all weight files (`model-00001-of-00004.safetensors`, etc.)
2. ✅ Reads only metadata (parameter names and locations) from all files
3. ✅ **Only loads assigned layers into memory** (e.g., layers 8-16 for stage 1)
4. ✅ Skips other layers using `PPMissingLayer` placeholders (zero memory cost)
5. ✅ Uses memory-mapped files for efficient access

**Example** (32-layer model, 4 stages):
- Stage 1 loads layers 8-16: ~2GB memory (not 8GB for full model!)
- Layers 0-7: Skipped (PPMissingLayer, no memory used)
- Layers 16-31: Skipped (PPMissingLayer, no memory used)

**Technical Details**:
- Uses safetensors' `safe_open()` for memory-mapped access
- Parameter filtering via `is_pp_missing_parameter()` in vLLM core
- Layer indices preserved (no renumbering needed)
- Same final memory footprint as pre-split models

### Run Pre-Split Pipeline with ZeroMQ (Simple Test)

For quick testing of pre-split models:

```bash
python tools/offline_pipelining/test_pipeline.py \
    --pipeline-dir /nfs_ssd/yanying/models/pipeline/Meta-Llama-3-8B/inference \
    --num-stages 4 \
    --test-input "Hello, world!" \
    --device cuda \
    --multiprocess
```

### Comparison: Pre-Split vs Dynamic Loading

| Feature | Pre-Split Models | Dynamic Loading (Full Model) |
|---------|-----------------|------------------------------|
| **Setup Required** | ✅ Run `serve_pipeline.py` once | ❌ None, use model directly |
| **Disk Space** | ❌ ~2x (original + split) | ✅ 1x (original only) |
| **Startup Time** | ✅ Fast (~10-15s) | ⚠️ Slightly slower (~15-25s) |
| **Memory Usage** | ✅ Minimal per stage | ✅ Minimal per stage (same) |
| **Configuration Flexibility** | ❌ Must re-export to change | ✅ Just change layer ranges |
| **Best For** | Production deployments | Development & prototyping |
| **Network Storage** | ✅ Minimal I/O | ⚠️ More file opens (metadata) |
| **Recommended When** | Fixed config, performance critical | Experimenting, shared storage |

**Bottom Line**: 
- Use **pre-split** for production (faster, battle-tested)
- Use **dynamic loading** for development and experimentation (flexible, no pre-processing)

### Validate Layer Range Configuration

Before launching pipeline stages, validate your layer range configuration:

```bash
# Validate explicit ranges
python tools/offline_pipelining/validate_layer_ranges.py \
    --total-layers 32 \
    --ranges "0-8" "8-16" "16-24" "24-32"

# Auto-calculate and validate from model
python tools/offline_pipelining/validate_layer_ranges.py \
    --model /path/to/Llama-3-8B \
    --num-stages 4

# Validate custom ranges against model
python tools/offline_pipelining/validate_layer_ranges.py \
    --model /path/to/Llama-3-8B \
    --ranges "0-10" "10-20" "20-32"
```

The validator checks:
- ✅ No overlapping layers between stages
- ✅ All layers covered exactly once (no gaps)
- ✅ Valid range formats and indices
- ✅ Ranges match total layer count

### Run Pipeline with External Pipeline Mode (Production Deployment)


For production use with external pipeline mode, you can use either:

**Option 1: HTTP Entry Point (Recommended for HTTP API access)**

```bash
python tools/offline_pipelining/http_index.py \
    --pipeline-dir /path/to/pipeline \
    --num-stages 4 \
    --base-port 15550 \
    --api-port 8000 \
    --gpu-ids 0,1,2,3 \
    --host 127.0.0.1 \
    --port 8080
```

This will:
- Launch all pipeline stages automatically
- Wait for stage 0 to be ready
- Expose an HTTP server on port 8080 (default) that proxies requests to stage 0
- Provide health check (`/health`) and pipeline status (`/pipeline/status`) endpoints
- Handle graceful shutdown and cleanup

**Key Design Principle**: Each stage is independent and only needs to know:
- Its own port (for binding PUSH sockets)
- Previous stage's address (for connecting PULL socket)
- Last stage's address (Stage 0 for return path)

**Option 2: Automated Test Script (Bind Mode)**

```bash
# Using the test script
bash tools/zeromq-pp-test.sh
```

The script uses **External Pipeline Mode with Bind Mode architecture**:
- **Bind Mode**: PUSH sockets bind to ports (senders), PULL sockets connect to addresses (receivers)
- This allows Kubernetes Service load balancing and flexible receiver connections
- Launches all pipeline stages as separate vLLM instances
- Configures ZeroMQ communication between stages using bind mode
- Maps each stage to a specific GPU
- Sets up forward and return communication paths
- Waits for initialization and runs a test inference
- Provides log files for each stage

**Configuration** (edit `tools/zeromq-pp-test.sh`):

```bash
BASE_MODEL_PATH="/home/yanying/pipeline_export/Llama-3-8B"  # Pipeline stages directory
NUM_STAGES=4                                                 # Number of stages
BASE_PORT=15550                                              # Starting ZeroMQ port
GPU_IDS=(0 1 2 3)                                           # GPU mapping (one per stage)
VLLM_SOURCE_DIR="/home/yanying/workspace/github/vllm"      # vLLM source directory
```

**Manual Launch** (Bind Mode - matches `zeromq-pp-test.sh` behavior):

**Bind Mode Architecture**:
- PUSH sockets bind to ports (senders bind, receivers connect)
- PULL sockets connect to Service addresses (receivers connect to senders)
- This allows Kubernetes Service load balancing

**Communication Flow**:
- Forward path: Stage i binds PUSH socket, Stage i+1 connects PULL socket to Stage i
- Return path: Last stage binds PUSH socket, Stage 0 connects PULL socket to last stage

**Parameter Names**:
- `--pipeline-prev-stage-service-addr`: Address of previous stage's PUSH Service (recommended, clear name)
  - Stage i+1 uses this to connect its PULL socket to Stage i's PUSH Service
  - In bind mode: Stage i binds PUSH, Stage i+1 connects PULL to it
- `--pipeline-next-stage-addr`: [DEPRECATED] Backward compatibility alias for `--pipeline-prev-stage-service-addr`
  - Despite the name, in bind mode this actually refers to the PREVIOUS stage's address
  - Kept for backward compatibility but the new name is recommended

**Key Principle**: Each stage only needs to know:
- Its own port (for binding PUSH socket if not last stage)
- Previous stage's address (for connecting PULL socket if not first stage)
- Last stage's address (Stage 0 for return path, if needed)

```bash
# Stage 0 (first stage with HTTP API)
# - Binds PUSH socket on port 15550 (Stage 1 will connect PULL to this)
# - Connects PULL socket to last stage's return port 15554 (if return path enabled)
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.cli.main serve \
    /path/to/pipeline/stage_0 \
    --pipeline-stage-mode external \
    --pipeline-stage-idx 0 \
    --pipeline-total-stages 4 \
    --pipeline-local-bind-port 15550 \
    --pipeline-prev-stage-addr 127.0.0.1:15554 \
    --tensor-parallel-size 1 \
    --port 8000

# Stage 1 (middle stage)
# - Connects PULL socket to Stage 0's PUSH Service (127.0.0.1:15550)
# - Binds PUSH socket on port 15550 (Stage 2 will connect PULL to this)
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.cli.main serve \
    /path/to/pipeline/stage_1 \
    --pipeline-stage-mode external \
    --pipeline-stage-idx 1 \
    --pipeline-total-stages 4 \
    --pipeline-local-bind-port 15550 \
    --pipeline-prev-stage-service-addr 127.0.0.1:15550 \
    --tensor-parallel-size 1 \
    --external-pp-worker

# Stage 2 (middle stage)
# - Connects PULL socket to Stage 1's PUSH Service (127.0.0.1:15550)
# - Binds PUSH socket on port 15550 (Stage 3 will connect PULL to this)
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.cli.main serve \
    /path/to/pipeline/stage_2 \
    --pipeline-stage-mode external \
    --pipeline-stage-idx 2 \
    --pipeline-total-stages 4 \
    --pipeline-local-bind-port 15550 \
    --pipeline-prev-stage-service-addr 127.0.0.1:15550 \
    --tensor-parallel-size 1 \
    --external-pp-worker

# Stage 3 (last stage)
# - Connects PULL socket to Stage 2's PUSH Service (127.0.0.1:15550)
# - Binds PUSH socket on port 15554 for return path (Stage 0 will connect PULL to this)
CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.cli.main serve \
    /path/to/pipeline/stage_3 \
    --pipeline-stage-mode external \
    --pipeline-stage-idx 3 \
    --pipeline-total-stages 4 \
    --pipeline-local-bind-port 15554 \
    --pipeline-prev-stage-service-addr 127.0.0.1:15550 \
    --tensor-parallel-size 1 \
    --external-pp-worker
```

**Port Configuration (Bind Mode)**:
- **Forward path**: 
  - Stage i (non-last) binds PUSH socket on port `BASE_PORT`
  - Stage i+1 connects PULL socket to `Stage_i_IP:BASE_PORT` (needs to know previous stage's address)
  - Note: All stages use the same `BASE_PORT` for forward path in the current implementation
- **Return path**:
  - Last stage binds PUSH socket on port `BASE_PORT + NUM_STAGES`
  - Stage 0 connects PULL socket to `Last_Stage_IP:BASE_PORT+NUM_STAGES` (needs to know last stage's address)
- Example with BASE_PORT=15550, NUM_STAGES=4:
  - Stage 0 binds PUSH on 15550, Stage 1 connects PULL to 127.0.0.1:15550
  - Stage 1 binds PUSH on 15550, Stage 2 connects PULL to 127.0.0.1:15550
  - Stage 2 binds PUSH on 15550, Stage 3 connects PULL to 127.0.0.1:15550
  - Stage 3 binds PUSH on 15554 (return), Stage 0 connects PULL to 127.0.0.1:15554

**Parameter Names**:
- `--pipeline-prev-stage-service-addr`: Address of previous stage's PUSH Service (recommended, clear name)
  - Used by Stage i+1 to connect its PULL socket to Stage i's PUSH Service
  - In bind mode: Stage i binds PUSH, Stage i+1 connects PULL to it
- `--pipeline-next-stage-addr`: [DEPRECATED] Backward compatibility alias
  - Despite the name, in bind mode this refers to the PREVIOUS stage's address
  - Kept for backward compatibility but the new name is recommended

**Design Principle**: Each stage is independent and only needs minimal information:
- **Stage 0**: Its own bind port (for PUSH), last stage return address (for return PULL)
- **Middle stages**: Their own bind port (for PUSH), previous stage address (for PULL)
- **Last stage**: Its own bind ports (forward PUSH and return PUSH), previous stage address (for PULL)

## Technical Details

### Tensor Transfer Process

1. **Sending Stage**:
   - Splits tensor dict into metadata and tensors
   - Serializes metadata with pickle
   - Moves tensors to CPU and serializes with `torch.save()`
   - Sends via ZeroMQ multipart messages

2. **Receiving Stage**:
   - Receives multipart messages
   - Deserializes metadata
   - Deserializes tensors and restores dtype
   - Moves tensors to target device (CPU or CUDA)
   - Reconstructs tensor dict

### Port Configuration

#### External Pipeline Mode with Bind Mode (`zeromq-pp-test.sh`, `http_index.py`, `test_pipeline.py`)
- **Architecture**: Bind Mode - PUSH sockets bind to ports, PULL sockets connect to addresses
- **Design Principle**: Each stage is independent and only needs minimal information about stages it connects to
- **Forward path ports**: 
  - Stage i (non-last) binds PUSH socket on `BASE_PORT` (all stages use same port in current implementation)
  - Stage i+1 connects PULL socket to previous stage's address (needs to know previous stage's IP:port)
- **Return path port**: 
  - Last stage binds PUSH socket on `BASE_PORT + NUM_STAGES` (e.g., 15554 for 4 stages)
  - Stage 0 connects PULL socket to last stage's address (needs to know last stage's IP:port)
- **Configuration**:
  - Configured in `tools/zeromq-pp-test.sh` via `BASE_PORT` variable
  - Configured in `http_index.py` via `--base-port` argument
- **Parameters**:
  - Uses `--pipeline-local-bind-port` for PUSH socket binding (each stage's own port)
  - Uses `--pipeline-prev-stage-service-addr` for connecting to previous stage's PUSH Service (IP:port, recommended)
    - Stage i+1 uses this to connect its PULL socket to Stage i's PUSH Service
    - In bind mode: Stage i binds PUSH, Stage i+1 connects PULL to it
  - Uses `--pipeline-next-stage-addr` as backward compatibility alias (deprecated, use `--pipeline-prev-stage-service-addr` instead)
  - Uses `--pipeline-prev-stage-addr` for Stage 0 to connect to last stage's return path (IP:port)

## Future Improvements

1. **Add Communication Backend Abstraction**: Implement Option 2 above
2. **GPU Direct Transfer**: Use CUDA IPC or GPU Direct RDMA for better performance
3. **Async Communication**: Overlap communication with computation
4. **Compression**: Add tensor compression for large hidden states
5. **Monitoring**: Add metrics for communication latency and throughput
6. **Error Recovery**: Add retry logic and better error handling
7. **Flow Control**: Implement backpressure for memory-constrained scenarios

## Related Commits

- `cbfc1eb`: Initial ZeroMQ implementation
- `83e77c4cb`: Added external pipeline mode with core vLLM modifications
- `cab0fe2c8`: Removed debug logs
- `7f9dca2b6`, `e326ee0f9`: Additional updates

## Summary: Core File Modifications

### Files Modified in Core vLLM (commit `83e77c4cb`)

These modifications are **required** for the "External Pipeline Mode" feature:

1. `vllm/v1/worker/gpu_model_runner.py` - Added external pipeline mode support (~1000 lines)
2. `vllm/distributed/device_communicators/zeromq_communicator.py` - New ZeroMQ communicator
3. `vllm/config/parallel.py` - Added external pipeline configuration options
4. `vllm/distributed/parallel_state.py` - External pipeline state management
5. `vllm/model_executor/models/utils.py` - External pipeline utilities
6. `vllm/model_executor/models/llama.py` - External pipeline layer support
7. `vllm/model_executor/models/transformers/base.py` - External pipeline base support
8. `vllm/entrypoints/cli/serve.py` - CLI arguments for external mode

### Files in Tools Directory

#### Files in `tools/offline_pipelining/`

1. `zmq_communicator.py` - ZeroMQ communication implementation
2. `test_pipeline.py` - Simple pipeline test (uses ZeroMQCommunicator directly)
3. `serve_pipeline.py` - Pipeline export tool
4. `test_model_zeromq.py` - Model validation test
5. `http_index.py` - HTTP entry point for ZeroMQ pipeline (launches stages using external pipeline mode and exposes HTTP API)

#### Files in `tools/` (External Pipeline Mode)

1. `zeromq-pp-test.sh` - Automated test script for external pipeline mode
   - **Mode**: External Pipeline Mode with **Bind Mode** architecture
   - **Architecture**: PUSH sockets bind to ports (senders), PULL sockets connect to addresses (receivers)
   - Launches multiple vLLM instances as pipeline stages
   - Configures ZeroMQ communication automatically using bind mode
   - Uses `--pipeline-local-bind-port` for PUSH socket binding
   - Uses `--pipeline-prev-stage-service-addr` for connecting to previous stage's PUSH Service (recommended)
  - Uses `--pipeline-next-stage-addr` as backward compatibility alias (deprecated)
  - Uses `--pipeline-prev-stage-addr` for Stage 0 to connect to last stage's return path
   - Handles GPU mapping and port configuration
   - Runs test inference and provides logging
   - **Requires**: External pipeline mode (core modifications from commit `83e77c4cb`)

## Troubleshooting

### Port Conflicts

If you see "Address already in use" errors:
- Check if ports are already in use: `netstat -tuln | grep <port>` or `ss -tuln | grep <port>`
- For external pipeline mode: Modify `BASE_PORT` in `tools/zeromq-pp-test.sh` or `--base-port` in `http_index.py`
- Ensure previous pipeline processes are terminated: `pkill -f "vllm.*serve"` or check PIDs from script output

### Timeout Errors

If communication times out:
- Increase `timeout_ms` in `ZeroMQCommunicator` initialization
- Check network connectivity between stages
- Verify all stages are running and connected
- For external pipeline mode: Check logs in `./zeromq_pp_logs/stage_*.log`
- Ensure stages are launched in order (wait for each stage to initialize before launching next)

### Tensor Shape Mismatches

If you see dtype or shape mismatches:
- Ensure all stages use the same model configuration
- Check that tensor metadata is correctly serialized/deserialized
- Verify device placement (CPU vs CUDA) is consistent
- For external pipeline mode: Verify layer ranges are correctly configured in each stage's `config.json`

### External Pipeline Mode Issues

If using `zeromq-pp-test.sh`:

1. **Stage fails to start**:
   - Check log file: `./zeromq_pp_logs/stage_<idx>.log`
   - Verify GPU availability: `nvidia-smi`
   - Ensure `VLLM_SOURCE_DIR` is set correctly if using source code
   - Check that stage directories exist and contain valid model files

2. **API server not responding**:
   - Wait longer for initialization (script waits 30 seconds, may need more for large models)
   - Check stage 0 log for API server startup messages
   - Verify port 8000 is not in use: `netstat -tuln | grep 8000`

3. **Communication errors between stages**:
   - Verify `LOCAL_IP` is correct (script auto-detects, but can be set via `VLLM_HOST_IP`)
   - Check that all stages can reach each other on the network
   - Verify port ranges don't conflict with other services
   - Check firewall settings if stages are on different machines

4. **GPU mapping issues**:
   - Ensure `GPU_IDS` array has at least `NUM_STAGES` entries
   - Verify GPUs are available: `nvidia-smi`
   - Check that each GPU has enough memory for the model stage

## Performance Considerations

- **CPU Transfer Overhead**: Current implementation transfers tensors via CPU. For better performance, consider GPU Direct transfer.
- **Serialization Cost**: Pickle and torch.save() add overhead. Consider using faster serialization formats.
- **Network Latency**: TCP sockets add latency. For local stages, consider using IPC sockets (`ipc://`).
- **Batch Size**: Larger batches amortize communication overhead better.

