# ZeroMQ Pipeline Parallelism Migration

## Overview

Starting from commit `cbfc1eb`, we migrated pipeline parallelism data transmission from NCCL to ZeroMQ. This change enables more flexible pipeline stage communication, especially for offline pipeline inference scenarios.

## Changes Summary

### Implementation Approaches

There are **two different approaches** to using ZeroMQ for pipeline parallelism:

1. **Monkey Patching Approach** (No core code changes required)
   - Used by `test_pipeline_engine.py`
   - Intercepts `get_pp_group()` calls via monkey patching
   - All ZeroMQ code is in `tools/offline_pipelining/` directory
   - **No modifications to vLLM core files**

2. **External Pipeline Mode** (Core code modifications)
   - Added in commit `83e77c4cb`
   - Integrated into vLLM core with new configuration options
   - Requires modifications to core vLLM files
   - More robust but requires maintaining fork

### Key Files Added/Modified

#### Tools Directory (No Core Changes Required)

1. **`tools/offline_pipelining/zmq_communicator.py`** (NEW)
   - Implements `ZeroMQCommunicator` class
   - Provides `send_tensor_dict()` and `recv_tensor_dict()` interfaces compatible with PyTorch distributed API
   - Uses ZeroMQ PUSH/PULL sockets for unidirectional communication between stages
   - Supports bidirectional communication for autoregressive generation (token ID feedback)

2. **`tools/offline_pipelining/test_pipeline.py`** (NEW)
   - Implements `PipelineStage` and `ZeroMQPipeline` classes
   - Uses `ZeroMQCommunicator` for inter-stage communication
   - Supports multiprocess pipeline execution

3. **`tools/offline_pipelining/test_pipeline_engine.py`** (NEW)
   - Adds `setup_zeromq_pp_group()` function that monkey-patches vLLM's `get_pp_group()`
   - Creates `ZeroMQPPGroup` wrapper to intercept pipeline communication calls
   - Allows using ZeroMQ with vLLM engine **without modifying core inference code**
   - Uses monkey patching to replace NCCL communication with ZeroMQ

4. **`tools/offline_pipelining/serve_pipeline.py`** (NEW)
   - Pipeline export tool for splitting models into stages
   - Exports each stage as HuggingFace format

5. **`tools/offline_pipelining/test_model_zeromq.py`** (NEW)
   - Test tool for validating pipeline stages with vLLM engine

6. **`tools/offline_pipelining/http_index.py`** (NEW)
   - HTTP entry point for ZeroMQ pipeline parallelism
   - Launches all pipeline stages using external pipeline mode
   - Exposes HTTP server that forwards inference requests to stage 0
   - Handles process lifecycle and cleanup
   - Provides health check and pipeline status endpoints

#### Core vLLM Files (Modified in commit `83e77c4cb`)

**Note**: These modifications are for the "External Pipeline Mode" feature. The monkey patching approach in `test_pipeline_engine.py` does **NOT** require these changes.

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
     - `pipeline_next_stage_addr`: Next stage address (IP:port)
     - `pipeline_prev_stage_addr`: Previous stage address
     - `pipeline_local_listen_port`: Local listen port

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

### ZeroMQ Socket Pattern

- **Forward direction**: PUSH/PULL sockets
  - Stage i binds PUSH socket to port `zmq_ports[i]`
  - Stage i+1 connects PULL socket to port `zmq_ports[i]`
  
- **Backward direction**: For autoregressive generation
  - Last stage sends token IDs back to first stage
  - Uses additional port range (base_port + 10000)

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

## Implementation Analysis: Two Approaches Compared

### Approach 1: Monkey Patching (No Core Changes)

**Used by**: `test_pipeline_engine.py`

This approach uses **monkey patching** to intercept communication without modifying model inference code:

1. **`test_pipeline_engine.py`** patches `vllm.distributed.parallel_state.get_pp_group()`:
   ```python
   def setup_zeromq_pp_group(stage_idx, num_stages, communicator):
       zmq_pp_group = ZeroMQPPGroup(stage_idx, num_stages, communicator)
       parallel_state_module._PP = zmq_pp_group
       parallel_state_module.get_pp_group = lambda: zmq_pp_group
   ```

2. **`ZeroMQPPGroup`** implements the same interface as NCCL-based PP groups, intercepting `send_tensor_dict()` and `recv_tensor_dict()` calls.

**Pros**:
- ✅ No modifications to vLLM core files
- ✅ Works with existing vLLM engine
- ✅ Easy to maintain (all code in tools directory)
- ✅ Can be used with any vLLM version (with minor adjustments)

**Cons**:
- ⚠️ Requires runtime patching
- ⚠️ May break with vLLM updates (need to update patching logic)
- ⚠️ Less integrated with vLLM's configuration system

### Approach 2: External Pipeline Mode (Core Modifications)

**Added in**: commit `83e77c4cb`

This approach integrates ZeroMQ support directly into vLLM core:

1. Added `pipeline_stage_mode` configuration option
2. Modified `gpu_model_runner.py` to handle external pipeline stages
3. Created `vllm/distributed/device_communicators/zeromq_communicator.py`
4. Added CLI arguments and configuration validation

**Pros**:
- ✅ Fully integrated with vLLM configuration system
- ✅ More robust and maintainable long-term
- ✅ Better error handling and validation
- ✅ Official support path

**Cons**:
- ⚠️ Requires maintaining a fork of vLLM
- ⚠️ Need to merge/rebase with upstream vLLM updates
- ⚠️ More complex to set up initially

### Which Approach to Use?

- **For offline pipeline testing/development**: Use **Approach 1 (Monkey Patching)** with `test_pipeline_engine.py`
- **For production deployment**: Consider **Approach 2 (External Pipeline Mode)** if you can maintain a vLLM fork

### Unnecessary Modifications Analysis

The following modifications in `gpu_model_runner.py` are **necessary** for external pipeline mode but **not required** for monkey patching approach:

1. **CPU tensor transfer** (line 3275): Required for ZeroMQ serialization
   - Comment: `# Move to CPU for ZeroMQ serialization`
   - **Necessary**: ZeroMQ requires CPU tensors for serialization

2. **Error handling for ZeroMQ deserialization** (lines 3593-3596): Helpful for debugging
   - **Necessary**: Catches deserialization issues early

3. **Comments mentioning ZeroMQ** (line 3744): Documentation only
   - **Optional**: Could be removed, but helpful for understanding

**Conclusion**: For the monkey patching approach, **no core vLLM modifications are needed**. All ZeroMQ communication is handled through the monkey-patched `get_pp_group()` interface.

## Usage

### Export Pipeline Stages

```bash
python tools/offline_pipelining/serve_pipeline.py \
    --model /path/to/model \
    --pipeline-parallel-size 4 \
    --output-dir /nfs_ssd/yanying/models/pipeline/Meta-Llama-3-8B/inference
```

### Run Pipeline with ZeroMQ

```bash
python tools/offline_pipelining/test_pipeline.py \
    --pipeline-dir /nfs_ssd/yanying/models/pipeline/Meta-Llama-3-8B/inference \
    --num-stages 4 \
    --test-input "Hello, world!" \
    --device cuda \
    --multiprocess
```

### Run Pipeline with vLLM Engine (ZeroMQ - Monkey Patching)

```bash
python tools/offline_pipelining/test_pipeline_engine.py \
    --pipeline-dir /path/to/pipeline \
    --num-stages 4 \
    --test-input "Hello, world!" \
    --device cuda
```

### Run Pipeline with External Pipeline Mode (ZeroMQ - Core Integration)

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

**Option 2: Automated Test Script**

```bash
# Using the test script
bash tools/zeromq-pp-test.sh
```

The script automatically:
- Launches all pipeline stages as separate vLLM instances
- Configures ZeroMQ communication between stages
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

**Manual Launch** (if you prefer to launch stages manually):

```bash
# Stage 0 (first stage with HTTP API)
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.cli.main serve \
    /path/to/pipeline/stage_0 \
    --pipeline-stage-mode external \
    --pipeline-stage-idx 0 \
    --pipeline-total-stages 4 \
    --pipeline-next-stage-addr 127.0.0.1:15550 \
    --pipeline-local-listen-port 15554 \
    --port 8000

# Stage 1 (middle stage)
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.cli.main serve \
    /path/to/pipeline/stage_1 \
    --pipeline-stage-mode external \
    --pipeline-stage-idx 1 \
    --pipeline-total-stages 4 \
    --pipeline-next-stage-addr 127.0.0.1:15551 \
    --pipeline-local-listen-port 15550 \
    --external-pp-worker

# Stage 2 (middle stage)
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.cli.main serve \
    /path/to/pipeline/stage_2 \
    --pipeline-stage-mode external \
    --pipeline-stage-idx 2 \
    --pipeline-total-stages 4 \
    --pipeline-next-stage-addr 127.0.0.1:15552 \
    --pipeline-local-listen-port 15551 \
    --external-pp-worker

# Stage 3 (last stage)
CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.cli.main serve \
    /path/to/pipeline/stage_3 \
    --pipeline-stage-mode external \
    --pipeline-stage-idx 3 \
    --pipeline-total-stages 4 \
    --pipeline-prev-stage-addr 127.0.0.1:15554 \
    --pipeline-local-listen-port 15552 \
    --external-pp-worker
```

**Port Configuration**:
- Forward path: `BASE_PORT + stage_idx` (stage_i sends to stage_{i+1})
- Return path: `BASE_PORT + NUM_STAGES` (last stage sends to stage 0)
- Stage 0 listens on return port to receive final results
- Non-first stages listen on forward ports to receive from previous stage

## Technical Details

### ZeroMQPPGroup Implementation

The `ZeroMQPPGroup` class in `test_pipeline_engine.py` provides a drop-in replacement for NCCL-based pipeline groups:

```python
class ZeroMQPPGroup:
    def send_tensor_dict(self, tensor_dict, dst=None, ...):
        """Forwards to ZeroMQCommunicator.send_tensor_dict()"""
        
    def recv_tensor_dict(self, src=None, ...):
        """Forwards to ZeroMQCommunicator.recv_tensor_dict()"""
```

This allows vLLM's model executor to use ZeroMQ without any code changes - it simply calls `get_pp_group().send_tensor_dict()` as usual, but the underlying implementation uses ZeroMQ instead of NCCL.

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

#### Monkey Patching Approach (`test_pipeline.py`, `test_pipeline_engine.py`)
- Default forward ports: `5555, 5556, 5557, ...` (one per stage boundary)
- Default backward port: `15555` (for token ID feedback)
- Can be customized via `--zmq-ports` argument

#### External Pipeline Mode (`zeromq-pp-test.sh`)
- Forward path ports: `BASE_PORT + stage_idx` (e.g., 15550, 15551, 15552 for 4 stages)
- Return path port: `BASE_PORT + NUM_STAGES` (e.g., 15554 for 4 stages)
- Configured in `tools/zeromq-pp-test.sh` via `BASE_PORT` variable
- Stage 0 listens on return port to receive final results from last stage
- Non-first stages listen on forward ports to receive from previous stage

## Future Improvements

1. **Add Communication Backend Abstraction**: Implement Option 2 above
2. **GPU Direct Transfer**: Use CUDA IPC or GPU Direct RDMA for better performance
3. **Async Communication**: Overlap communication with computation
4. **Compression**: Add tensor compression for large hidden states
5. **Monitoring**: Add metrics for communication latency and throughput
6. **Error Recovery**: Add retry logic and better error handling
7. **Flow Control**: Implement backpressure for memory-constrained scenarios

## Related Commits

- `cbfc1eb`: Initial ZeroMQ implementation (monkey patching approach)
- `83e77c4cb`: Added external pipeline mode with core vLLM modifications
- `cab0fe2c8`: Removed debug logs
- `7f9dca2b6`, `e326ee0f9`: Additional updates

## Summary: Core File Modifications

### Files Modified in Core vLLM (commit `83e77c4cb`)

These modifications are **only needed** for the "External Pipeline Mode" feature. The monkey patching approach (`test_pipeline_engine.py`) does **NOT** require these changes:

1. `vllm/v1/worker/gpu_model_runner.py` - Added external pipeline mode support (~1000 lines)
2. `vllm/distributed/device_communicators/zeromq_communicator.py` - New ZeroMQ communicator
3. `vllm/config/parallel.py` - Added external pipeline configuration options
4. `vllm/distributed/parallel_state.py` - External pipeline state management
5. `vllm/model_executor/models/utils.py` - External pipeline utilities
6. `vllm/model_executor/models/llama.py` - External pipeline layer support
7. `vllm/model_executor/models/transformers/base.py` - External pipeline base support
8. `vllm/entrypoints/cli/serve.py` - CLI arguments for external mode

### Files in Tools Directory

#### Files in `tools/offline_pipelining/` (No Core Changes Required)

All files in `tools/offline_pipelining/` work with **unmodified vLLM core** using monkey patching:

1. `zmq_communicator.py` - ZeroMQ communication implementation
2. `test_pipeline.py` - Simple pipeline test (no vLLM engine)
3. `test_pipeline_engine.py` - Pipeline test with vLLM engine (uses monkey patching)
4. `serve_pipeline.py` - Pipeline export tool
5. `test_model_zeromq.py` - Model validation test
6. `http_index.py` - HTTP entry point for ZeroMQ pipeline (launches stages and exposes HTTP API)

#### Files in `tools/` (External Pipeline Mode)

1. `zeromq-pp-test.sh` - Automated test script for external pipeline mode
   - Launches multiple vLLM instances as pipeline stages
   - Configures ZeroMQ communication automatically
   - Handles GPU mapping and port configuration
   - Runs test inference and provides logging
   - **Requires**: External pipeline mode (core modifications from commit `83e77c4cb`)

## Troubleshooting

### Port Conflicts

If you see "Address already in use" errors:
- Check if ports are already in use: `netstat -tuln | grep <port>` or `ss -tuln | grep <port>`
- For monkey patching approach: Use `--zmq-ports` to specify different ports
- For external pipeline mode: Modify `BASE_PORT` in `tools/zeromq-pp-test.sh`
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

