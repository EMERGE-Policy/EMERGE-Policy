# LIBERO Evaluation

Run all commands from the repository root.

Episode driver configs use `libero.bddl_root` plus a relative
`libero.bddl_file_name`. The evaluator fills both from the selected benchmark
task, so the base evaluation JSON does not need a fixed scene. LIBERO-Plus and
LIBERO-Pro share this config writer; Plus virtual filenames retain their
parameter suffixes, and Pro resolves each suite's actual data root, including
generated Environment data exposed through symlinks.

## 1. Start services (terminal 1)

### WAM (default)

```bash
WAM_GPU=0 \
VGGT_GPU=1 \
SAM3_GPU=2 \
bash scripts/model_server/start_external_model_servers.sh --services cosmos,vggt,sam3
```

### VLA

```bash
OPENPI_GPU=0 \
VGGT_GPU=1 \
SAM3_GPU=2 \
bash scripts/model_server/start_external_model_servers.sh --services openpi,vggt,sam3
```

Batch size defaults to 1 for all services.
See [service settings and health checks](../model_server/README.md).

## 2. Run evaluation (terminal 2)

```bash
conda activate EmergePolicy
```

### WAM: one task, one trial

```bash
python scripts/Libero_eval/eval_libero_agent.py \
  --policy-backend wam \
  --suite libero_object \
  --task-ids 0 \
  --trials-per-task 1 \
  --workers 1 \
  --record-video \
  --output-dir artifacts/libero_agent_eval/wam_smoke
```

### VLA: one task, one trial

```bash
python scripts/Libero_eval/eval_libero_agent.py \
  --policy-backend vla \
  --suite libero_object \
  --task-ids 0 \
  --trials-per-task 1 \
  --workers 1 \
  --record-video \
  --output-dir artifacts/libero_agent_eval/vla_smoke
```

For a short execution check, add `--max-steps 16`.
For task enumeration only, add `--dry-run`.

### Multiple workers

Start the selected service stack with
[batch size 4](../model_server/README.md#optional-batching-for-multiple-workers),
then run:

```bash
python scripts/Libero_eval/eval_libero_agent.py \
  --policy-backend wam \
  --suite libero_object \
  --task-ids all \
  --trials-per-task 1 \
  --workers 4 \
  --record-video \
  --output-dir artifacts/libero_agent_eval/wam_libero_object
```

For VLA, use `--policy-backend vla` and a separate `--output-dir`.

### Full evaluation

```bash
python scripts/Libero_eval/eval_libero_agent.py \
  --policy-backend wam \
  --full \
  --workers 4 \
  --record-video \
  --output-dir artifacts/libero_agent_eval/wam_full
```

`--full`: four suites × ten tasks × fifty trials. For VLA, change the backend
to `vla` and use a separate output directory.

### Resume

Repeat the original command with `--resume`. Keep the same backend, profile,
task selection, and output directory.

## Options

| Option | Usage |
| --- | --- |
| `--policy-backend` | `wam` (default) or `vla` |
| `--suite` | `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`; repeat or use `all` |
| `--task-ids` | `0`, `0,2,3`, `2-5`, or `all` |
| `--trials-per-task` | Trials per task |
| `--workers` | Concurrent environments |
| `--wam-conditioning-mode` | `task` (default), `phase`, or `task_with_phase` |
| `--profile-path` | Custom profile |
| `--max-steps` | Maximum action steps per episode |
| `--dry-run` | List tasks without simulation |
| `--record-video` | Save `rollout.mp4` |
| `--no-web-video` | Skip H.264 conversion |
| `--stream` | Live video at `http://127.0.0.1:8008/` |
| `--resume` | Resume the same output directory |
| `--continue-on-error` | Continue after infrastructure errors |

Default profiles: WAM uses `robot/profiles/libero_wam_mujoco.md`;
VLA uses `robot/profiles/libero_mujoco.md`.
For WAM phase conditioning, follow the
[T5 setup](../model_server/README.md#optional-wam-phase-conditioning).

When using SSH, forward port `8008` for `--stream`.

## Output

- `evaluation_plan.json`: run configuration.
- `results.jsonl`: episode results.
- `summary.json`: success rates.
- Per attempt: `result.json`, `status.json`, `agent.log`, `watchdog.log`,
  `driver_config.json`, `workspace/`, and optional `rollout.mp4`.
