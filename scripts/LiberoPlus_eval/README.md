# Emerge-Policy LIBERO-Plus Evaluation Guide

Run all commands from the repository root:
`cd /path/to/EMERGE-Policy`

## 0. Prerequisites

Prepare the following before running an evaluation:

- The LIBERO-Plus source must be under `third_party/libero_plus/`. The source
  checkout does not include the large asset bundle. Download `assets.zip` from
  the official [Sylvest/LIBERO-plus Hugging Face dataset](https://huggingface.co/datasets/Sylvest/LIBERO-plus)
  and extract it into `third_party/libero_plus/libero/libero/`, as described in
  the root [installation guide](../../README.md#install-the-libero-plus-assets).
  The resulting `third_party/libero_plus/libero/libero/assets/` directory must
  contain the LIBERO-Plus assets.
- ImageMagick must be installed in the isolated
  `third_party/imagemagick_env/.env/` directory. Install it from the repository
  root with `bash third_party/imagemagick_env/install.sh`. The evaluator configures
  `MAGICK_HOME` and `LD_LIBRARY_PATH` automatically, but the directory must
  already exist.

See [environment installation](../../README.md#installation) in the
root README for the setup command. These dependencies are loaded even when
tasks are enumerated with `--dry-run`.

## 1. Start services (terminal 1)

### WAM (default)

Use `cache-then-online` because the Language Instructions dimension can contain
text that is not present in the official LIBERO T5 cache.

```bash
WAM_GPU=0 \
VGGT_GPU=1 \
SAM3_GPU=2 \
WAM_EMBEDDING_MODE=cache-then-online \
bash scripts/model_server/start_external_model_servers.sh --services cosmos,vggt,sam3
```

### VLA

```bash
OPENPI_GPU=0 \
VGGT_GPU=1 \
SAM3_GPU=2 \
bash scripts/model_server/start_external_model_servers.sh --services openpi,vggt,sam3
```

Batch size defaults to 1 for all services. For multiple workers, see
[batch settings](../model_server/README.md#optional-batching-for-multiple-workers).
Health checks are in the [service guide](../model_server/README.md#2-check-services).

## 2. Run the evaluation (terminal 2)

```bash
conda activate EmergePolicy
cd /path/to/EMERGE-Policy
```

### Check task selection first

`--dry-run` only enumerates tasks and creates the evaluation plan; it does not
start simulation:

```bash
python scripts/LiberoPlus_eval/eval_libero_plus_agent.py \
  --policy-backend wam \
  --dimension sensor_noise \
  --count 4 \
  --dry-run
```

### WAM: evaluate one robustness dimension

```bash
python scripts/LiberoPlus_eval/eval_libero_plus_agent.py \
  --policy-backend wam \
  --dimension sensor_noise \
  --count 4 \
  --trials-per-task 1 \
  --workers 4 \
  --record-video \
  --output-dir artifacts/libero_plus_agent_eval/sensor_noise
```

For VLA, use `--policy-backend vla` and a separate `--output-dir`.

`--dimension` accepts an official name or one of the following snake-case
names:

| Snake-case name | Official name |
| --- | --- |
| `camera_viewpoints` | Camera Viewpoints |
| `robot_initial_states` | Robot Initial States |
| `language_instructions` | Language Instructions |
| `light_conditions` | Light Conditions |
| `background_textures` | Background Textures |
| `sensor_noise` | Sensor Noise |
| `objects_layout` | Objects Layout |

`--count N` is shorthand for `--episodes-per-dimension N` and selects exactly
N episodes for each dimension. Selection is determined by `--seed` and
distributed round-robin across the four standard suites. LIBERO-Plus only
supports dimension-based task selection; do not pass the standard LIBERO
options `--suite`, `--task-ids`, or `--full`.

Each perturbed task is run once, so `--trials-per-task` must be `1`.

### Evaluate multiple dimensions

Repeat `--dimension` to select their union:

```bash
python scripts/LiberoPlus_eval/eval_libero_plus_agent.py \
  --policy-backend wam \
  --dimension camera_viewpoints \
  --dimension objects_layout \
  --count 4 \
  --workers 4 \
  --output-dir artifacts/libero_plus_agent_eval/camera_and_layout
```

Use `--dimension all` to select all seven dimensions. For example, `--count 4`
runs `7 × 4 = 28` episodes in total.

### Resume an interrupted evaluation

Append `--resume` to the original command and keep the same dimensions, count,
seed, trial, and `--output-dir`. The evaluator validates task selection against
`evaluation_plan.json` and skips completed episodes.

## 3. Recording and browser viewing

- Add `--record-video` to generate `rollout.mp4` for every attempt. Videos are
  converted to H.264 by default; use `--no-web-video` to disable conversion.
- Add `--stream` to combine the concurrent worker feeds into a live browser
  grid.

```bash
python scripts/LiberoPlus_eval/eval_libero_plus_agent.py \
  --policy-backend wam \
  --dimension light_conditions \
  --count 4 \
  --workers 8 --stream --record-video \
  --output-dir artifacts/libero_plus_agent_eval/light_conditions_stream
```

The evaluator prints the viewing address, for example
`http://127.0.0.1:8008/`. When running through SSH and VS Code, forward port
`8008` in the **PORTS** panel and open it in your local browser.

Streaming options are `--stream-host` (default: `127.0.0.1`), `--stream-port`
(default: `8008`), and `--stream-fps` (default: `10`).

## Common options

| Option | Description |
| --- | --- |
| `--dimension` | Required robustness dimension; repeat it or use `all` |
| `--episodes-per-dimension` / `--count` | Required number of episodes per dimension |
| `--trials-per-task` | Must be `1` |
| `--workers` | Concurrent episode count |
| `--policy-backend` | `wam` (default) or `vla` |
| `--wam-conditioning-mode` | `task` (default), `phase`, or `task_with_phase` |
| `--seed` | Task-selection seed; default: `7` |
| `--watchdog-ready-timeout-s` | MuJoCo startup timeout; Plus default: `600` seconds |
| `--dry-run` | Enumerate tasks and create the plan without simulation |
| `--record-video` | Record the agent-view video |
| `--no-web-video` | Disable automatic H.264 conversion |
| `--stream` | View multiple live feeds in a browser |
| `--resume` | Resume using the same `--output-dir` |
| `--continue-on-error` | Continue scheduling after an `infrastructure_error` |
| `--driver-config` | Driver configuration; default: `dev/libero_plus_eval.json` |
| `--output-dir` | Output directory; use a new directory for a new evaluation |

WAM reuses `robot/profiles/libero_wam_mujoco.md`; VLA uses
`robot/profiles/libero_mujoco.md`. Use `--profile-path` only to override the
selected profile explicitly.

## Output files

The output directory contains:

- `evaluation_plan.json`: dimensions, selected tasks, and seed for the run.
- `results.jsonl`: one line per episode and the primary resume record.
- `summary.json`: overall success rates grouped by dimension and suite.
- `libero_plus_grid.json` and `libero_plus_grid.md`: the result grid for all
  seven robustness dimensions.

Each dimension directory contains `selection.json`, `summary.json`, and attempt
directories grouped by suite. Every attempt contains `result.json`,
`status.json`, `agent.log`, `watchdog.log`, `driver_config.json`, optional
`rollout.mp4`, and `workspace/`.

> Run both the external model services and the evaluation inside `tmux` to
> prevent an SSH disconnect from terminating the processes.
