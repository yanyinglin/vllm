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
--pipeline-next-stage-addr <IP>:<PORT>      # Address of next stage (sends data forward)
--pipeline-local-listen-port <PORT>        # Port to listen for return path (receives final results from last stage)
--port <API_PORT>                          # HTTP API port (only stage 0 needs this)
```

### Example

```bash
python -m vllm.entrypoints.cli.main serve \
    /path/to/pipeline/stage_0 \
    --pipeline-stage-mode external \
    --pipeline-stage-idx 0 \
    --pipeline-total-stages 4 \
    --pipeline-next-stage-addr 127.0.0.1:15550 \
    --pipeline-local-listen-port 15554 \
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
--pipeline-next-stage-addr <IP>:<PORT>      # Address of next stage
--pipeline-local-listen-port <PORT>        # Port to listen for previous stage (receives data)
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
--pipeline-prev-stage-addr <IP>:<PORT>      # Return path address to stage 0 (sends final results)
--pipeline-local-listen-port <PORT>        # Port to listen for previous stage (receives data)
--external-pp-worker                        # Mark as external PP worker
```

### Example

```bash
python -m vllm.entrypoints.cli.main serve \
    /path/to/pipeline/stage_3 \
    --pipeline-stage-mode external \
    --pipeline-stage-idx 3 \
    --pipeline-total-stages 4 \
    --pipeline-prev-stage-addr 127.0.0.1:15554 \
    --pipeline-local-listen-port 15552 \
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
    --pipeline-next-stage-addr 127.0.0.1:15550 \
    --pipeline-local-listen-port 15554 \
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
    --pipeline-next-stage-addr 127.0.0.1:15551 \
    --pipeline-local-listen-port 15550 \
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
    --pipeline-next-stage-addr 127.0.0.1:15552 \
    --pipeline-local-listen-port 15551 \
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
    --pipeline-prev-stage-addr 127.0.0.1:15554 \
    --pipeline-local-listen-port 15552 \
    --tensor-parallel-size 1 \
    --external-pp-worker
```

## Port Configuration Rules

### Forward Path (Stage i → Stage i+1)

- Port formula: `BASE_PORT + stage_idx`
- Example with BASE_PORT=15550:
  - Stage 0 → Stage 1: `15550` (BASE_PORT + 0)
  - Stage 1 → Stage 2: `15551` (BASE_PORT + 1)
  - Stage 2 → Stage 3: `15552` (BASE_PORT + 2)

### Return Path (Last Stage → Stage 0)

- Port formula: `BASE_PORT + NUM_STAGES`
- Example with BASE_PORT=15550, NUM_STAGES=4:
  - Stage 3 → Stage 0: `15554` (BASE_PORT + 4)

### Port Assignment Summary

| Stage | Forward Listen Port | Next Stage Address | Return Listen Port | Return Address |
|-------|---------------------|-------------------|-------------------|----------------|
| 0     | -                   | 127.0.0.1:15550  | 15554             | -              |
| 1     | 15550               | 127.0.0.1:15551  | -                 | -              |
| 2     | 15551               | 127.0.0.1:15552  | -                 | -              |
| 3     | 15552               | -                 | -                 | 127.0.0.1:15554 |

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
- [ ] `--pipeline-next-stage-addr <IP>:<PORT>`
- [ ] `--pipeline-local-listen-port <RETURN_PORT>`
- [ ] `--port <API_PORT>`
- [ ] `--tensor-parallel-size 1`

### Middle Stage Checklist
- [ ] `--pipeline-stage-mode external`
- [ ] `--pipeline-stage-idx <i>`
- [ ] `--pipeline-total-stages <N>`
- [ ] `--pipeline-next-stage-addr <IP>:<PORT>`
- [ ] `--pipeline-local-listen-port <FORWARD_PORT>`
- [ ] `--external-pp-worker`
- [ ] `--tensor-parallel-size 1`

### Last Stage Checklist
- [ ] `--pipeline-stage-mode external`
- [ ] `--pipeline-stage-idx <N-1>`
- [ ] `--pipeline-total-stages <N>`
- [ ] `--pipeline-prev-stage-addr <IP>:<RETURN_PORT>`
- [ ] `--pipeline-local-listen-port <FORWARD_PORT>`
- [ ] `--external-pp-worker`
- [ ] `--tensor-parallel-size 1`

