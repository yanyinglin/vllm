# vLLM ZeroMQ Pipeline Parallelism Startup Parameters

This document describes the required parameters to start vLLM with ZeroMQ-based external pipeline parallelism.

## Required Parameters (All Stages)

All pipeline stages require these base parameters:

```bash
--pipeline-stage-mode external          # Enable external pipeline mode
--pipeline-stage-idx <0-based-index>    # Current stage index (0, 1, 2, ...)
--pipeline-total-stages <total>         # Total number of stages
--tensor-parallel-size 1                # Tensor parallel size (must be 1 for external mode)
```

## Stage 0 (First Stage) Parameters

The first stage exposes the HTTP API and receives final results from the last stage.

### Required Parameters

```bash
--pipeline-stage-mode external
--pipeline-stage-idx 0
--pipeline-total-stages <total>
--pipeline-local-bind-port <PORT>           # Port to bind PUSH socket (for Stage 1 to connect PULL)
--pipeline-prev-stage-addr <IP>:<PORT>     # Address of last stage's return PUSH Service (for return path PULL socket to connect)
--port <API_PORT>                          # HTTP API port (only stage 0 needs this)
```

### Example

```bash
python -m vllm.entrypoints.cli.main serve \
    /path/to/pipeline/stage_0 \
    --pipeline-stage-mode external \
    --pipeline-stage-idx 0 \
    --pipeline-total-stages 4 \
    --pipeline-local-bind-port 15550 \
    --pipeline-prev-stage-addr 127.0.0.1:15554 \
    --tensor-parallel-size 1 \
    --port 8000
```

## Middle Stages (Stage 1 to N-2) Parameters

Middle stages receive data from the previous stage and forward to the next stage.

### Required Parameters

```bash
--pipeline-stage-mode external
--pipeline-stage-idx <index>
--pipeline-total-stages <total>
--pipeline-prev-stage-service-addr <IP>:<PORT>  # Address of previous stage's PUSH Service (for PULL socket to connect in bind mode)
--pipeline-local-bind-port <PORT>          # Port to bind PUSH socket (for next stage to connect PULL)
--external-pp-worker                        # Mark as external PP worker (no HTTP API)
```

### Example

```bash
python -m vllm.entrypoints.cli.main serve \
    /path/to/pipeline/stage_1 \
    --pipeline-stage-mode external \
    --pipeline-stage-idx 1 \
    --pipeline-total-stages 4 \
    --pipeline-next-stage-addr 127.0.0.1:15551 \
    --pipeline-local-listen-port 15550 \
    --tensor-parallel-size 1 \
    --external-pp-worker
```

## Last Stage (Stage N-1) Parameters

The last stage receives data from the previous stage and sends final results back to stage 0.

### Required Parameters

```bash
--pipeline-stage-mode external
--pipeline-stage-idx <last-index>
--pipeline-total-stages <total>
--pipeline-prev-stage-service-addr <IP>:<PORT>  # Address of previous stage's PUSH Service (for PULL socket to connect)
--pipeline-local-bind-port <RETURN_PORT>       # Port to bind return PUSH socket (for Stage 0 to connect return PULL)
--external-pp-worker                            # Mark as external PP worker
```

### Example

```bash
python -m vllm.entrypoints.cli.main serve \
    /path/to/pipeline/stage_3 \
    --pipeline-stage-mode external \
    --pipeline-stage-idx 3 \
    --pipeline-total-stages 4 \
    --pipeline-prev-stage-service-addr 127.0.0.1:15550 \
    --pipeline-local-bind-port 15554 \
    --tensor-parallel-size 1 \
    --external-pp-worker
```

## Complete Example: 4-Stage Pipeline

### Stage 0

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.cli.main serve \
    /path/to/pipeline/stage_0 \
    --pipeline-stage-mode external \
    --pipeline-stage-idx 0 \
    --pipeline-total-stages 4 \
    --pipeline-local-bind-port 15550 \
    --pipeline-prev-stage-addr 127.0.0.1:15554 \
    --tensor-parallel-size 1 \
    --port 8000
```

### Stage 1

```bash
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.cli.main serve \
    /path/to/pipeline/stage_1 \
    --pipeline-stage-mode external \
    --pipeline-stage-idx 1 \
    --pipeline-total-stages 4 \
    --pipeline-prev-stage-service-addr 127.0.0.1:15550 \
    --pipeline-local-bind-port 15550 \
    --tensor-parallel-size 1 \
    --external-pp-worker
```

### Stage 2

```bash
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.cli.main serve \
    /path/to/pipeline/stage_2 \
    --pipeline-stage-mode external \
    --pipeline-stage-idx 2 \
    --pipeline-total-stages 4 \
    --pipeline-prev-stage-service-addr 127.0.0.1:15550 \
    --pipeline-local-bind-port 15550 \
    --tensor-parallel-size 1 \
    --external-pp-worker
```

### Stage 3 (Last)

```bash
CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.cli.main serve \
    /path/to/pipeline/stage_3 \
    --pipeline-stage-mode external \
    --pipeline-stage-idx 3 \
    --pipeline-total-stages 4 \
    --pipeline-prev-stage-service-addr 127.0.0.1:15550 \
    --pipeline-local-bind-port 15554 \
    --tensor-parallel-size 1 \
    --external-pp-worker
```

## Port Configuration Rules (Bind Mode)

### Forward Path (Stage i → Stage i+1)

- All stages bind PUSH socket on the same port: `BASE_PORT`
- Stage i+1 connects PULL socket to Stage i's address: `Stage_i_IP:BASE_PORT`
- Example with BASE_PORT=15550:
  - Stage 0 binds PUSH on 15550, Stage 1 connects PULL to 127.0.0.1:15550
  - Stage 1 binds PUSH on 15550, Stage 2 connects PULL to 127.0.0.1:15550
  - Stage 2 binds PUSH on 15550, Stage 3 connects PULL to 127.0.0.1:15550

### Return Path (Last Stage → Stage 0)

- Port formula: `BASE_PORT + NUM_STAGES`
- Example with BASE_PORT=15550, NUM_STAGES=4:
  - Stage 3 binds return PUSH on 15554, Stage 0 connects return PULL to 127.0.0.1:15554

### Port Assignment Summary

| Stage | Forward Bind Port (PUSH) | Previous Stage Address (PULL connects) | Return Bind Port (PUSH) | Return Address (PULL connects) |
|-------|---------------------------|----------------------------------------|-------------------------|-------------------------------|
| 0     | 15550                     | -                                      | -                       | 127.0.0.1:15554               |
| 1     | 15550                     | 127.0.0.1:15550                        | -                       | -                              |
| 2     | 15550                     | 127.0.0.1:15550                        | -                       | -                              |
| 3     | -                         | 127.0.0.1:15550                        | 15554                   | -                              |

## Optional Parameters

```bash
--pipeline-layer-range "0-8"           # Manually specify layer range (usually auto-detected from config.json)
--enforce-eager                        # Disable CUDA graphs (useful for debugging)
--dtype float16                        # Model data type
--trust-remote-code                    # Trust remote code
--max-model-len 4096                   # Maximum model length
```

## Environment Variables

For using vLLM source code directly:

```bash
export VLLM_SOURCE_DIR="/path/to/vllm/source"
export PYTHONPATH="${VLLM_SOURCE_DIR}:${PYTHONPATH}"
```

Then use:
```bash
python -m vllm.entrypoints.cli.main serve [args...]
```

Instead of:
```bash
vllm serve [args...]
```

## Important Notes

1. **All stages must use the same `--pipeline-total-stages` value**
2. **`--tensor-parallel-size` must be 1** (external mode limitation)
3. **Only stage 0 needs `--port`** (provides HTTP API)
4. **All non-first stages must use `--external-pp-worker`**
5. **Port configuration must be correct** - ensure stages can communicate
6. **Launch stages in order** - wait for each stage to initialize before launching the next
7. **Layer ranges are auto-detected** from `_pipeline_info` in each stage's `config.json`

## Quick Reference

### Stage 0 Checklist
- [ ] `--pipeline-stage-mode external`
- [ ] `--pipeline-stage-idx 0`
- [ ] `--pipeline-total-stages <N>`
- [ ] `--pipeline-local-bind-port <PORT>` (for forward PUSH)
- [ ] `--pipeline-prev-stage-addr <IP>:<PORT>` (for return PULL, if return path enabled)
- [ ] `--port <API_PORT>`
- [ ] `--tensor-parallel-size 1`

### Middle Stage Checklist
- [ ] `--pipeline-stage-mode external`
- [ ] `--pipeline-stage-idx <i>`
- [ ] `--pipeline-total-stages <N>`
- [ ] `--pipeline-prev-stage-service-addr <IP>:<PORT>` (previous stage's PUSH Service)
- [ ] `--pipeline-local-bind-port <PORT>` (for forward PUSH)
- [ ] `--external-pp-worker`
- [ ] `--tensor-parallel-size 1`

### Last Stage Checklist
- [ ] `--pipeline-stage-mode external`
- [ ] `--pipeline-stage-idx <N-1>`
- [ ] `--pipeline-total-stages <N>`
- [ ] `--pipeline-prev-stage-service-addr <IP>:<PORT>` (previous stage's PUSH Service)
- [ ] `--pipeline-local-bind-port <RETURN_PORT>` (for return PUSH, if return path enabled)
- [ ] `--external-pp-worker`
- [ ] `--tensor-parallel-size 1`

