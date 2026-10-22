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

### Full EMERGE + Robometer: 40 tasks, one trial each

Use spatial, object, goal and LIBERO-10: ten tasks each, **40 episodes total**.
LIBERO-90 is excluded. This runs the complete EMERGE agent (LLM planning,
perception and robot tools), using four Cosmos candidates scored by Robometer.
The existing agent/API configuration is required. No training is involved.

Start the official Robometer service in a separate terminal and keep it running:

```bash
cd /private/Emerge-Policy
ROBOMETER_REPO=/private/robometer ROBOMETER_GPU=3 \
ROBOMETER_MODEL_PATH=/private/Emerge-Policy/checkpoints/robometer-4b-fft-libero \
bash scripts/model_server/start_robometer_server.sh
```

Start Cosmos and both perception services in another terminal. The 40 task
instructions are already in the supplied T5 cache; online T5 is unnecessary.

```bash
cd /private/Emerge-Policy
WAM_PYTHON=/private/cosmos-policy/.venv/bin/python \
WAM_PYTHONPATH=/private/cosmos-policy WAM_GPU=4 \
PERCEPTION_ENV=EmergePolicy VGGT_GPU=1 SAM3_GPU=2 \
WAM_EMBEDDING_MODE=cache-only \
ROBOMETER_URL=http://127.0.0.1:8100 \
ROBOMETER_MODEL_PATH=/private/Emerge-Policy/checkpoints/robometer-4b-fft-libero \
bash scripts/model_server/start_external_model_servers.sh --services cosmos,vggt,sam3
```

If Cosmos is already running with these settings, start only the missing
services, e.g. `--services vggt,sam3`, rather than starting a second Cosmos.
Wait for model loading to finish. Robometer is an official HTTP service;
it does not participate in EMERGE WebSocket discovery. The evaluator checks
its explicit URL and checkpoint identity, plus Cosmos, VGGT and SAM3 readiness,
before starting any episode. Its URL/model path must match the Cosmos settings.
A model RPC failure during an episode stops further scheduling by default.

Run the fixed 40-episode entrypoint in the EmergePolicy environment:

```bash
cd /private/Emerge-Policy
conda activate EmergePolicy
bash scripts/Libero_eval/run_robometer_40.sh
```

It defaults to two workers, seed 7, one trial per task and video recording.
`EVAL_WORKERS`, `EVAL_OUTPUT_DIR`, `EMERGE_PYTHON` and `EVAL_EPISODE_TIMEOUT`
can override the corresponding execution settings. Add `--dry-run` to list
the 40 tasks without starting episodes; add `--resume` to skip completed ones.
Do not add `--full`: that selects 50 trials per task. Default output:
`artifacts/emerge_robometer_40/`.

Robometer mode keeps each environment's candidates in a separate Cosmos
request. Workers overlap agent/perception waits; Cosmos does not combine
different environments in its native future-image decoding batch.

For a run that survives closing a terminal, keep the model services alive too
(e.g. in tmux), and launch the evaluator with a log:

```bash
mkdir -p artifacts/emerge_robometer_40
nohup bash scripts/Libero_eval/run_robometer_40.sh \
  > artifacts/emerge_robometer_40/run.log 2>&1 < /dev/null &
```

The Web UI is not needed for batch evaluation: both entrypoints launch
`robot.controller` and `Emerge.cli.headless` using the same backend. An older
Web conversation using `vla_execute`/OpenPI does not verify the separate
Cosmos-to-Robometer connection. `Connection refused` at port 8100 means the
Robometer server is unavailable at the configured address. A discovery error
such as `Expected one vggt endpoint, found []` means VGGT is unavailable.

Each completed attempt saves a separate video at:

```text
artifacts/emerge_robometer_40/<suite>/task_XX/trial_00_seed_7/attempt_<timestamp>/rollout.mp4
```

`result.json` and `results.jsonl` include the absolute `video_path`. The MP4
is finalized after the episode closes; it may not play while being written.
Install `ffmpeg` for the default H.264 conversion (Ubuntu: `apt-get install -y
ffmpeg`); without it OpenCV's mp4v can be unreadable in VSCode/Chromium. An older
run without `--record-video` has no rollout video to retrieve retroactively.

One trial gives a 0%/100% result for each task. Each suite's rate is successes/10,
and the complete overall rate is successes/40. Inspect completion counts and
infrastructure errors before interpreting success rates. Current EMERGE action
limits are spatial 620, object 680, goal 700 and LIBERO-10 920, including the
existing recovery allowance. A short `--max-steps 16` check is not a full task
success benchmark. To compare `joint_value`, run the generic evaluator with
`--suite all --task-ids all --trials-per-task 1 --record-video`, use
`dev/libero_agent_eval.json` and a separate output directory, keeping other
settings identical.

LIBERO-90 remains opt-in via the generic evaluator's `--suite libero_90`.
It is not selected by this entrypoint; most of its task instructions are absent
from the supplied T5 cache, so it needs a separate online-T5 setup.

### Resume

Repeat the original command with `--resume`. Keep the same backend, profile,
task selection, and output directory. Completed task outcomes are skipped;
episodes ending in `infrastructure_error:` are retried with the same task,
seed and initial state in a new attempt directory. Previous attempts remain
on disk, and summaries use the latest result for each episode key.

## Options

| Option | Usage |
| --- | --- |
| `--policy-backend` | `wam` (default) or `vla` |
| `--suite` | `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`; repeat; `all` selects the original four |
| `--task-ids` | `0`, `0,2,3`, `2-5`, or `all` |
| `--trials-per-task` | Trials per task |
| `--workers` | Concurrent environments |
| `--wam-conditioning-mode` | `task` (default), `phase`, or `task_with_phase` |
| `--profile-path` | Custom profile |
| `--max-steps` | Maximum action steps per episode |
| `--dry-run` | List tasks without simulation |
| `--robometer-url` / `--robometer-model-path` | Explicit official HTTP URL and checkpoint identity; also read ROBOMETER_URL / ROBOMETER_MODEL_PATH |
| `--record-video` | Save `rollout.mp4` and its `video_path` in results |
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

Live progress and a rough remaining-time estimate (use the same worker count
as the evaluator):

```bash
python scripts/Libero_eval/monitor_libero_eval.py \
  artifacts/emerge_robometer_40 --workers 2 --watch 10
```

This is read-only and does not launch models or call the LLM. It shows completed
episodes, action steps in active episodes, elapsed time since plan preparation
or resume, and estimated remaining time from mean completed-episode duration
divided by the worker count. Without completed episodes the estimate is unknown;
early estimates and shared-model queueing can vary substantially. `Ctrl+C` in
the monitor terminal stops only the monitor. Detailed execution remains in
each attempt's `agent.log`, `watchdog.log` and `status.json`.

- `evaluation_plan.json`: run configuration.
- `results.jsonl`: episode results.
- `summary.json`: success rates.
- Per attempt: `result.json`, `status.json`, `agent.log`, `watchdog.log`,
  `driver_config.json`, `workspace/`, and optional `rollout.mp4`.
