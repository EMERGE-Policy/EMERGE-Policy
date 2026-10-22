# Robometer scoring for Cosmos WAM

`wam.search.score_mode: robometer` keeps Cosmos candidate action generation and
its joint future-state prediction. The independent official Robometer HTTP
server scores each candidate future; the existing Emerge planner chooses the
highest score. `none`, `joint_value`, and `q_value` remain available.

The score is **last-frame task progress**. It is a candidate ranking heuristic,
not a calibrated success probability, expected return, or Q value. Cosmos may
still generate a value latent, but its value is never used in this mode.

## Verified upstream contracts

- EMERGE's pinned Cosmos commit: `18a2accadf4e7a3531e56754102af5a24d2316da`.
  `cosmos_policy/experiments/robot/cosmos_utils.py:get_action` returns
  `all_camera_images` (processed wrist then primary) and, with
  `generate_future_state_and_value_in_parallel=True`, a
  `future_image_predictions` dictionary containing `future_image` and
  `future_wrist_image`. These are action-chunk endpoint views, not a rollout.
- Official Robometer commit: `352d160389daa964788de1ec933d1925f3a6de4f`.
  [eval_server.py](https://github.com/robometer/robometer/blob/352d160389daa964788de1ec933d1925f3a6de4f/robometer/evals/eval_server.py)
  owns the persistent model, using `utils/save.py:load_model_from_hf` and
  `utils/setup_utils.py:setup_batch_collator`. The original checkpoint heads,
  tokenizer, processor and output decoding are used; EMERGE defines none of them.
- The lightweight client mirrors `make_progress_sample`,
  `build_multipart_payload` and `post_evaluate_batch_npy` in
  [scripts/example_inference.py](https://github.com/robometer/robometer/blob/352d160389daa964788de1ec933d1925f3a6de4f/scripts/example_inference.py).
  That example imports matplotlib at module import time, so EMERGE keeps a
  minimal NumPy/HTTP multipart envelope instead of importing the example.
  It posts to **`/evaluate_batch_npy`**, with `sample_0`, `sample_1`, etc.,
  `.npy` uploads and `{"__numpy_file__": ...}` references. The official server
  reconstructs samples in numerical sample order; progress rows have the same
  order. The client checks candidate indices and exact output counts.

Default checkpoint: `jesbu1/robometer-4b-fft-libero`. Alternative:
`robometer/Robometer-4B`, or a local checkpoint directory. Both published
configs use Qwen3-VL-4B, a trained progress head, multi-image input and per-frame
progress tokens. The LIBERO checkpoint index includes the six progress-head
weight/bias tensors. The official loader restores those checkpoint weights.

## Input and output

Each of N candidates sends one RGB `uint8` array `(2,H,W,3)`:

1. The current image returned by Cosmos's **official preprocessing**.
2. This candidate's decoded future image from the **same camera**.

Default view is primary/third-person; `ROBOMETER_VIEW=wrist` selects the wrist
for both frames. Camera views are never interleaved as temporal frames. There
is no stored history, so environments and resets cannot share stale frames.
Unexecuted predictions never enter real observation state.

The executor sends raw LIBERO observation images. Cosmos `_prepare_request`
applies its configured vertical flip (default on). Upstream `get_action` then
performs deterministic JPEG compression, resize and 90%-area center crop,
and provides the exact processed current images. The future decode has the
same orientation and geometry. Robometer's LIBERO dataset loader also uses
vertical flipping of raw LIBERO images. The client therefore **does not flip,
crop, resize, convert BGR, or interpolate** either processed image again.
Changing Cosmos preprocessing changes the scoring distribution as well.

`use_frame_steps=false` is intentional: the official frame-step path creates
four resampled/repeated frames per prefix. The verified checkpoint's normal
multi-image collator preserves the two image tokens and uses the official
processor (`do_sample_frames=false`), so no artificial intermediate motion is
needed. Its trained-on-eight-frame setting does not require padding this
multi-image inference request to eight frames. The client requires exactly
two finite progress values for each candidate and takes only the second.
Official discrete-head decoding remains in the service.

`/model_info` is checked on every scoring request: its `model_path` must match
the configured path, and its config must support a trained progress head,
multi-image input, per-frame progress tokens and no frame shuffling. This
prevents accidentally scoring through an incompatible server/checkpoint.
No client-side random scoring head is initialized.

## Install official Robometer in a separate environment

Run from a terminal, using Python 3.10 as required by the pinned official
`pyproject.toml`. Do not install this dependency set into EmergePolicy or Cosmos.

```bash
conda create -n robometer python=3.10 -y
conda activate robometer
python -m pip install uv

git clone https://github.com/robometer/robometer.git /private/robometer
git -C /private/robometer checkout 352d160389daa964788de1ec933d1925f3a6de4f
cd /private/robometer
export UV_PROJECT_ENVIRONMENT=/opt/robometer/.venv
export ROBOMETER_PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
uv sync --python "$CONDA_PREFIX/bin/python" --extra robometer

# Multipart uploads are required by the official HTTP endpoint.
# Keep torchao compatible with the upstream torch==2.8.0 requirement.
uv pip install --python "$ROBOMETER_PYTHON" \
  'python-multipart>=0.0.20' 'torchao==0.13.0' 'setuptools<81'
```

The official dependency set is large. Its model loader may also fetch the
Qwen3-VL base checkpoint and processor even when model_path points to local
Robometer weights. Populate the Hugging Face cache for offline operation.

Download the default weights to a directory of your choice:

```bash
export ROBOMETER_MODEL_PATH=/private/model/robometer-4b-fft-libero
"$UV_PROJECT_ENVIRONMENT/bin/hf" download jesbu1/robometer-4b-fft-libero \
  --local-dir "$ROBOMETER_MODEL_PATH"
```

For the general checkpoint, change the repository argument to
`robometer/Robometer-4B`. Alternatively, set `ROBOMETER_MODEL_PATH` to a Hub ID
and let the official loader download it on first startup. A local directory
must include the official `config.yaml` and complete weights/shards.

### Precision compatibility for the verified environment

The unmodified upstream server, Torch 2.8.0, Transformers 4.57.6 and Unsloth
2026.10.4 encountered a float32/bfloat16 mismatch inside Qwen's first attention
projection and missing hidden-state outputs in the optimized Qwen path. Use
the **existing official `model.torch_dtype` and `model.use_unsloth` parameters**
in the local checkpoint config for this setup:

```bash
"$ROBOMETER_PYTHON" - <<'PY'
import os
from pathlib import Path
import yaml

path = Path(os.environ['ROBOMETER_MODEL_PATH']) / 'config.yaml'
backup = path.with_name('config.original.yaml')
if not backup.exists():
    backup.write_text(path.read_text())
config = yaml.safe_load(path.read_text())
config['model']['torch_dtype'] = 'float32'
config['model']['use_unsloth'] = False
path.write_text(yaml.safe_dump(config, sort_keys=False))
PY
```

This selects upstream's standard Transformers loading path and FP32 precision;
learned weights, heads, tokenizer, preprocessing, inference code and the
official HTTP server remain upstream.
FP32 requires more memory; use a GPU with enough capacity (validation uses a
46 GB L20). Restore `config.original.yaml` to `config.yaml` to try the published
bfloat16 setting with a different compatible environment. A Hub-ID startup
uses the published config; for this precision override use a local model path.
Scoring metadata includes the server's actual `inference_dtype`.

## Start the official service

From the EMERGE repository, with the same `ROBOMETER_MODEL_PATH` used below:

```bash
cd /private/Emerge-Policy
export ROBOMETER_REPO=/private/robometer
export ROBOMETER_PYTHON=/opt/robometer/.venv/bin/python
export ROBOMETER_MODEL_PATH=/private/model/robometer-4b-fft-libero
export ROBOMETER_GPU=3
bash scripts/model_server/start_robometer_server.sh
```

The launcher executes the unmodified official `robometer.evals.eval_server`
with its Hydra config, one visible GPU and one worker. The model loads during
service initialization and is reused for candidates and subsequent requests.
Defaults: port 8100, host 127.0.0.1. Override `ROBOMETER_PORT`,
`ROBOMETER_HOST`, or `ROBOMETER_PYTHON` if needed.
The environment is kept on the container's regular filesystem because copying
large dependencies to this `/private` mount produced incomplete dependency
metadata. The source and downloaded Robometer checkpoint remain in `/private`.
The unused failed copy is isolated under
`/private/robometer/.venv_failed_mount_copy`; delayed entries on the mount
prevented complete removal. It is never used by the running service.

```bash
curl --noproxy '*' http://127.0.0.1:8100/health
curl --noproxy '*' http://127.0.0.1:8100/model_info
```

Robometer does **not** implement EMERGE's model-discovery handshake or
`/healthz`. It has an explicit HTTP URL; `/health` in the EMERGE console will
not discover it. Starting the official service is a separate command.

## Start Cosmos and select the new mode

Ensure Cosmos and perception environments are installed as in the root README.
The Cosmos environment needs the light HTTP client as well as its transport:

```bash
conda activate cosmos-policy
python -m pip install 'httpx>=0.28,<1' 'websockets>=16,<17' 'msgpack>=1.1,<2'

cd /private/Emerge-Policy
ROBOMETER_URL=http://127.0.0.1:8100 \
ROBOMETER_MODEL_PATH=/private/model/robometer-4b-fft-libero \
ROBOMETER_TIMEOUT=120 \
ROBOMETER_VIEW=primary \
WAM_GPU=0 VGGT_GPU=1 SAM3_GPU=2 \
bash scripts/model_server/start_external_model_servers.sh --services cosmos,vggt,sam3
```

These map to Cosmos `--robometer-url`, `--robometer-model-path`,
`--robometer-timeout`, and `--robometer-view`. The path string must match the
official server's `model_path`; it is an identity check, not a second model
load in Cosmos. Direct Cosmos startup supports the same arguments.

Run one LIBERO task with the provided config:

```bash
conda activate EmergePolicy
cd /private/Emerge-Policy
python scripts/Libero_eval/eval_libero_agent.py \
  --policy-backend wam --suite libero_object --task-ids 0 \
  --trials-per-task 1 --workers 1 --record-video \
  --driver-config dev/libero_robometer_eval.json \
  --robometer-url http://127.0.0.1:8100 \
  --robometer-model-path /private/model/robometer-4b-fft-libero \
  --output-dir artifacts/libero_robometer_smoke
```

Add `--max-steps 16` for a bounded pipeline smoke test (not a task-success
evaluation). The new JSON preserves the standard driver settings, selects
`wam.search: {num_candidates: 4, score_mode: robometer}` and increases the
WAM client timeout to 600 seconds because candidate VAE decoding and external
scoring take additional time. Agent/API configuration is required as in the
existing evaluator. This is inference, not a training workflow.

The evaluator now checks Cosmos, VGGT, SAM3 and the explicit Robometer HTTP
URL/model identity before launching episodes. Keep all services running for
the duration of evaluation. For four ten-task suites, one trial per task and
videos, use the [40-task entrypoint](../Libero_eval/README.md#full-emerge--robometer-40-tasks-one-trial-each).

To switch back to the original Cosmos joint-value ranking, use the existing
`--driver-config dev/libero_agent_eval.json` with a new output directory.
No Robometer call is made for `joint_value`, even if Cosmos has a Robometer
URL configured. `none` and opt-in `q_value` keep their original behavior.

## Batching, diagnostics and failures

The public Cosmos runtime can still accept multiple concurrent environments,
but Robometer-mode requests get singleton scheduler keys. Each environment
generates its N candidates sequentially, then scores them in **one HTTP batch**.
`WAM_MAX_BATCH_SIZE>1` continues to benefit other modes; it will not batch
Robometer environments together. Direct multi-environment `infer_batch`
calls containing this mode explicitly fail. The existing native batch helper
does not decode futures and is deliberately unchanged.

Response `scoring` metadata records backend, model ID/path, score semantics,
camera, frame count and elapsed scoring time. Cosmos logs candidate scores;
the planner logs the chosen index and carries the metadata in search decisions.
The normal wire format still carries actions and scores, not large future images.
No video/image dumping is enabled by this integration.

Timeouts, HTTP failures, model/config mismatch, missing futures, wrong image
layout/dtype, empty/wrong-length outputs and non-finite values stop the request
with explicit errors. There is no fallback to Cosmos scores or zero scores and
no automatic replay of failed HTTP inference. As with other local GPU work,
timing out a client cannot forcibly interrupt already-running server kernels.

## Verification

```bash
conda activate EmergePolicy
cd /private/Emerge-Policy
python -m unittest dev.verify_robometer -v
```

These are **mock tests**: real multipart bytes, candidate-dependent future
images, terminal-progress selection, runtime/planner integration, original
joint-value regression, timeout/HTTP/model/output failures and singleton
batch routing. They do not demonstrate real model inference.

To test the real model with saved LIBERO images, supply a uint8 `.npy` of two
ordered RGB frames from one camera. For multiple `--frames` files, all first
frames must be identical and each second frame must be its candidate future.

```bash
python -m scripts.model_server.check_robometer \
  --url http://127.0.0.1:8100 \
  --model-path /private/model/robometer-4b-fft-libero \
  --task 'pick up the cream cheese and place it in the basket' \
  --frames /path/to/libero_two_frames.npy
```

This is a **real-model interface test** only when connected to the real official
server. Full Cosmos/LIBERO verification uses the evaluator command above.
Two-frame predicted-state progress can rank candidates, but its effectiveness
under model-generated futures and short context requires benchmark evaluation.

For a bounded end-to-end WAM check without an outer LLM/API call, use the
provided real-model smoke script. It creates a separate LIBERO environment,
uses official initial states, generates/scores three candidates and executes
16 actions with the existing executor. It asserts highest-score selection and
records backend provenance in `result.json`; `task_success` is reported
separately from completing the bounded action chunk.

```bash
python -m dev.smoke_robometer_wam \
  --candidates 3 --steps 16 --score-mode robometer \
  --output-dir artifacts/robometer_validation/robometer

python -m dev.smoke_robometer_wam \
  --candidates 3 --steps 16 --score-mode joint_value \
  --output-dir artifacts/robometer_validation/joint_value
```

Use `--cosmos-port` if it is not 8003, or `--libero-source` for a different
official LIBERO checkout. `--frames-only` saves two real observation frames
to `libero_two_frames.npy` for the HTTP-only check above. EGL device settings
must match the available render devices; this container's validation uses
`MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0` for simulation, with model
services on separate GPUs.

### Results from this implementation session (2026-10-14)

The following results use the real default LIBERO-finetuned weights, unmodified
official source at the revision above, `model.use_unsloth=false` and
`model.torch_dtype=float32`. Cosmos uses the repository's pinned upstream source
and existing LIBERO checkpoints. The task is `libero_object`, task 0:
"pick up the alphabet soup and place it in the basket".

| Check | Observed result |
| --- | --- |
| Official service, real saved LIBERO two-frame input | final progress `0.02421047`, HTTP 200, 2.57 seconds |
| Real Cosmos → Robometer → planner → MuJoCo, N=3 | scores `[0.11637270, 0.12364490, 0.12163898]`, selected index **1**, 16 actions executed, scoring 4.55 seconds |
| Real original joint_value comparison, same task/initial state/seeds | scores `[0.27036545, 0.27019662, 0.27110770]`, selected index **2**, 16 actions executed |
| Published BF16/Unsloth service incompatibility | actual HTTP 500 propagated through WAM; **zero** actions executed, no fallback |
| Mock interface/regression/error checks | 10 tests passed |
| Syntax and patch checks | Python compileall, Bash syntax and git diff --check passed |

Saved evidence in this workspace:
`artifacts/robometer_validation/real_frames/interface_result.json`,
`artifacts/robometer_validation/robometer/result.json`, and
`artifacts/robometer_validation/joint_value/result.json`.
After preparing the final `/private/robometer` checkout and `/opt/robometer/.venv`
environment, the repeated real smoke test produced the same three scores and
selected index 1 (`artifacts/robometer_validation/handoff_smoke/result.json`).
These are bounded pipeline checks: **task_success=false** after 16 actions in
both modes. No complete-episode success-rate claim is made. The general
`robometer/Robometer-4B` config was inspected but those weights were not run;
wrist selection and unsupported multi-environment batching were covered by
mock tests. Ranking quality with only an action-chunk endpoint remains to be
measured on full benchmark episodes.

### Commands for the prepared workspace

The official source is installed at `/private/robometer`, the independent
environment is `/opt/robometer/.venv`, and the downloaded LIBERO checkpoint is
`/private/Emerge-Policy/checkpoints/robometer-4b-fft-libero`. Its local config
already sets the two compatibility parameters above.

Terminal 1 (Robometer):

```bash
cd /private/Emerge-Policy
ROBOMETER_REPO=/private/robometer \
ROBOMETER_PYTHON=/opt/robometer/.venv/bin/python \
ROBOMETER_MODEL_PATH=/private/Emerge-Policy/checkpoints/robometer-4b-fft-libero \
ROBOMETER_GPU=3 bash scripts/model_server/start_robometer_server.sh
```

Terminal 2 (reuse this workspace's existing Cosmos environment/checkouts):

```bash
cd /private/Emerge-Policy
WAM_PYTHON=/private/cosmos-policy/.venv/bin/python \
WAM_PYTHONPATH=/private/cosmos-policy WAM_GPU=4 \
ROBOMETER_URL=http://127.0.0.1:8100 \
ROBOMETER_MODEL_PATH=/private/Emerge-Policy/checkpoints/robometer-4b-fft-libero \
bash scripts/model_server/start_external_model_servers.sh --services cosmos
```

Terminal 3:

```bash
cd /private/Emerge-Policy
MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 \
/opt/conda/envs/EmergePolicy/bin/python -m dev.smoke_robometer_wam \
  --score-mode robometer --candidates 3 --steps 16 \
  --output-dir artifacts/robometer_validation/new_smoke
```

Use `--score-mode joint_value` and a different output directory for the original
ranking. Do not start duplicate processes if 8100/8003 are already running.
For the full agent evaluator, also start VGGT/SAM3 as in the main guide.
