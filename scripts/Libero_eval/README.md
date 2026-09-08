# Emerge-Policy LIBERO Evaluation Guide

Run all commands from the repository root:
`cd /data/yuqingchi/Code/Emerge-Policy`

## 1. Start the three external model services (terminal 1)

Use the following recommended configuration to start the OpenPI, VGGT, and
SAM3 services required for LIBERO evaluation:

```bash
OPENPI_MAX_BATCH_SIZE=4 \
OPENPI_BATCH_WAIT_MS=10 \
VGGT_MAX_BATCH_SIZE=4 \
VGGT_BATCH_WAIT_MS=10 \
SAM3_MAX_BATCH_SIZE=4 \
SAM3_BATCH_WAIT_MS=10 \
bash scripts/model_server/start_external_model_servers.sh --services openpi,vggt,sam3
```

For service selection, GPU assignment, dynamic batching, ports, model paths,
health checks, and the complete launcher parameter reference, see the
[External model servers guide](../model_server/README.md).

## 2. Run the evaluation (terminal 2)

```bash
conda activate EmergePolicy
cd /data/yuqingchi/Code/Emerge-Policy
```

### Evaluate one suite (all tasks, one trial per task)

```bash
python scripts/Libero_eval/eval_libero_agent.py \
  --suite libero_object \
  --task-ids all \
  --trials-per-task 1 \
  --workers 4 \
  --record-video \
  --output-dir artifacts/libero_agent_eval/libero_object
```

Valid `--suite` values are `libero_spatial`, `libero_object`, `libero_goal`,
and `libero_10`.

### Evaluate selected tasks

```bash
python scripts/Libero_eval/eval_libero_agent.py \
  --suite libero_object \
  --task-ids 0 \
  --trials-per-task 50 \
  --workers 4 \
  --output-dir artifacts/libero_agent_eval/libero_object_task_00
```

`--task-ids` accepts one ID such as `0`, a comma-separated list such as
`0,2,3`, a range such as `2-5`, or `all`.

### Evaluate all four suites (2,000 episodes)

```bash
python scripts/Libero_eval/eval_libero_agent.py \
  --full \
  --workers 4 \
  --record-video \
  --output-dir artifacts/libero_agent_eval/full
```

`--full` runs 4 suites × 10 tasks × 50 trials with seed 7. Do not add
`--full` when running only one suite.

### Resume an interrupted evaluation

Append `--resume` to the original command and keep the same `--output-dir`.
Completed episodes are skipped regardless of whether they succeeded or
failed.

## 3. Recording and browser viewing

- Add `--record-video` to generate `rollout.mp4` for every attempt. Videos are
  converted to H.264 by default for direct playback in a browser or VS Code.
  Add `--no-web-video` to disable conversion and retain the original mp4v
  encoding.
- Add `--stream` to combine the concurrent worker feeds into a live browser
  grid. The layout is chosen from `--workers`, for example 8→3×3 and 4→2×2.

```bash
python scripts/Libero_eval/eval_libero_agent.py \
  --suite libero_goal --task-ids all \
  --workers 8 --stream --record-video \
  --output-dir artifacts/libero_agent_eval/libero_goal_stream
```

The launcher prints the viewing address, for example
`http://127.0.0.1:8008/`. When using **SSH with VS Code**, forward port `8008`
from the **PORTS** panel and open it in your local browser.

Streaming options are `--stream-host` (default: `127.0.0.1`), `--stream-port`
(default: `8008`), and `--stream-fps` (default: `10`). Set the host to
`0.0.0.0` only when access from another machine is required, because doing so
exposes the port externally.

## Common options

| Option | Description |
| --- | --- |
| `--suite` | Suite name; may be repeated, or set to `all` |
| `--task-ids` | `all`, `0,2,3`, or a range such as `2-5` |
| `--trials-per-task` | Number of evaluation trials per task |
| `--workers` | Number of concurrent episodes; starting with 3–4 is recommended |
| `--full` | Run all four suites with 50 trials per task |
| `--record-video` | Record the agent-view video |
| `--no-web-video` | Disable automatic H.264 conversion |
| `--stream` | View multiple live feeds in a browser |
| `--resume` | Resume using the same `--output-dir` |
| `--continue-on-error` | Continue subsequent episodes after an `infrastructure_error` |
| `--output-dir` | Output directory; use a new directory for a new evaluation |

## Output files

Top-level files are `results.jsonl` (one line per episode and the primary
resume record) and `summary.json` (success-rate summary).

Each attempt directory contains `result.json`, `status.json`, `agent.log`,
`watchdog.log`, `driver_config.json`, optional `rollout.mp4`, and `workspace/`
with `ROBOT_STATE.md`, `ACTION.md`, and `PLAN.md`.

> Run both the three external model services and the evaluation inside `tmux`
> to prevent an SSH disconnect from terminating the processes.
