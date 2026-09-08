# External model servers

Heavyweight models run as persistent WebSocket services:

| Service | Port | Environment | Health check |
|---|---:|---|---|
| OpenPI | 8000 | `pi05_server` | `http://localhost:8000/healthz` |
| VGGT | 8001 | `EmergePolicy` | `http://localhost:8001/healthz` |
| SAM3 | 8002 | `EmergePolicy` | `http://localhost:8002/healthz` |
| Cosmos Policy | 8003 | `cosmos-policy` | `http://localhost:8003/healthz` |

Recommended default command:

```bash
OPENPI_GPU=0,1,2 \
OPENPI_MAX_BATCH_SIZE=4 \
OPENPI_BATCH_WAIT_MS=10 \
VGGT_GPU=3 \
VGGT_MAX_BATCH_SIZE=4 \
VGGT_BATCH_WAIT_MS=10 \
SAM3_GPU=4 \
SAM3_MAX_BATCH_SIZE=4 \
SAM3_BATCH_WAIT_MS=10 \
bash scripts/model_server/start_external_model_servers.sh --services openpi,vggt,sam3
```

Select services with `--services` or the `SERVICES` environment variable. The
accepted names are `openpi`, `vggt`, `sam3`, `cosmos`, and `all`:

```bash
bash scripts/model_server/start_external_model_servers.sh --services openpi,vggt,sam3

bash scripts/model_server/start_external_model_servers.sh --services cosmos,vggt,sam3

SERVICES=all bash scripts/model_server/start_external_model_servers.sh
```

The backend aliases are symmetric: `vla` selects `openpi`, and `wam` selects
`cosmos`.

Set `OPENPI_PYTHON` to an existing environment's Python executable when Conda
is unavailable, analogous to `WAM_PYTHON`.

## Dynamic batching

OpenPI, VGGT, and SAM3 support server-side dynamic batching. Each service
collects concurrent requests until the batch reaches its configured maximum
or the collection window expires. The WebSocket request and response formats
do not change.

| Service | Maximum batch variable | Collection-window variable | Default |
|---|---|---|---|
| OpenPI | `OPENPI_MAX_BATCH_SIZE` | `OPENPI_BATCH_WAIT_MS` | `4` / `10 ms` |
| VGGT | `VGGT_MAX_BATCH_SIZE` | `VGGT_BATCH_WAIT_MS` | `4` / `0 ms` |
| SAM3 | `SAM3_MAX_BATCH_SIZE` | `SAM3_BATCH_WAIT_MS` | `4` / `0 ms` |

For example, enable batching for all three services:

```bash
OPENPI_MAX_BATCH_SIZE=4 \
OPENPI_BATCH_WAIT_MS=10 \
VGGT_MAX_BATCH_SIZE=4 \
VGGT_BATCH_WAIT_MS=10 \
SAM3_MAX_BATCH_SIZE=4 \
SAM3_BATCH_WAIT_MS=10 \
bash scripts/model_server/start_external_model_servers.sh --services openpi,vggt,sam3
```

The two settings have the following meanings:

- `*_MAX_BATCH_SIZE` must be at least `1` and limits the number of requests in
  one inference call.
- `*_BATCH_WAIT_MS` must be non-negative and sets the maximum time spent
  collecting a batch after its first request arrives. A full batch runs
  immediately without waiting for the deadline.

Only compatible inputs can share a batch:

- OpenPI requests need the same data-tree structure and stack-compatible field
  shapes.
- VGGT requests need the same number of views and RGB image shapes.
- SAM3 requests need the same per-view image shapes, target count, and
  `max_instances` value.

Dynamic batching requires concurrent requests. A client that waits for each
response before sending its next request will normally observe a batch size of
`1`. Larger batches can improve throughput under concurrency but consume more
GPU memory. A longer collection window can fill batches more reliably but adds
queueing latency at low concurrency. For strictly single-request execution,
set the maximum batch size to `1` and the collection window to `0`.

OpenPI reports `batch_size`, `padded_batch_size`, `queue_ms`, and `infer_ms` in
the response's `server_timing` field. VGGT and SAM3 report
`dynamic_batching`, `max_batch_size`, and `batch_wait_ms` in the first metadata
message sent after connection. Cosmos Policy WAM does not currently use these
dynamic-batching settings.

OpenPI, VGGT, SAM3, and WAM share the NumPy/msgpack encoding and `/healthz`
implementation in `external_model_server/protocol.py`. The ndarray wire format remains compatible
with `openpi_client.msgpack_numpy`.

Connections first receive a metadata message. Inference responses have either
`{"ok": true, "result": ...}` or `{"ok": false, "error": ...}`.

## Cosmos Policy WAM

Cosmos Policy WAM uses a separate environment and model paths. Configure them
when selecting `cosmos` or its `wam` backend alias:

```bash
WAM_ENV=cosmos-policy \
WAM_GPU=0 \
WAM_POLICY_CHECKPOINT=/absolute/path/to/policy.ckpt \
WAM_BASE_MODEL_DIR=/absolute/path/to/predict2-base \
WAM_DATASET_STATS=/absolute/path/to/dataset_stats.json \
WAM_T5_EMBEDDINGS=/absolute/path/to/t5_embeddings.pkl \
WAM_CONFIG_FILE=cosmos_policy/config/config.py \
WAM_PYTHONPATH="$PWD/third_party/cosmos-policy" \
  bash scripts/model_server/start_external_model_servers.sh --services cosmos,vggt,sam3
```

If Conda is unavailable but the environment already exists, set its Python
executable directly instead of `WAM_ENV`:

```bash
WAM_PYTHON=/absolute/path/to/cosmos-policy/bin/python \
bash scripts/model_server/start_external_model_servers.sh --services cosmos
```

The launcher also uses an available `python3` or `python` for port checks. Set
`PORT_CHECK_PYTHON` explicitly on hosts where neither is on `PATH`.

The server uses the shared WebSocket/msgpack protocol, serves the common
`/healthz` endpoint, and keeps the versioned WAM schema inside the message
payload. Cosmos Policy generates and scores candidates; Emerge planner and
search select the candidate; Controller alone executes the resulting action chunk.
Keep the host bound to loopback unless an authenticated transport is added.

VGGT and SAM3 are required when evaluation enables object location or
perception-led recovery. They are not required for a direct policy-only WAM
request.

By default WAM remains cache-only. For phase-local conditioning, enable the
Cosmos server's in-process T5 encoder and cache-first online fallback:

```bash
WAM_PYTHON=/absolute/path/to/cosmos-policy/bin/python \
WAM_EMBEDDING_MODE=cache-then-online \
WAM_GENERATED_T5_CACHE=/writable/path/generated_t5.pkl \
T5_CACHE_DIR=/persistent/huggingface/cache \
  bash scripts/model_server/start_external_model_servers.sh --services cosmos
```

Pin `T5_MODEL_REVISION` for reproducible runs. The WAM request keeps
`task_instruction` and `phase_instruction` separate and selects `task`,
`phase`, or `task_with_phase` conditioning.

## VGGT request

```python
{
    "reference_view": "camera_1",
    "views": [
        {
            "name": "camera_1",
            "rgb": rgb_uint8,
            "intrinsics": intrinsics_float64,
            "T_world_camera": transform_float64,
        }
    ],
}
```

The metric backend requires at least two calibrated views. Its response contains
the processed RGB images, metric depth, confidence, world point maps, camera
diagnostics, and alignment diagnostics.

## SAM3 request

```python
{
    "views": [{"name": "camera_1", "rgb": rgb_uint8}],
    "targets": [
        {"object_key": "orange_juice_1", "prompt": "yellow juice carton"}
    ],
}
```

The response contains one mask result per view, including confidence, bounding
box, mask array, and mask area for every target.

## Launcher parameter reference

Run the following command to inspect the available settings, defaults, and
descriptions in the terminal:

```bash
bash scripts/model_server/start_external_model_servers.sh --help
```

Except for `--services`, all settings below are environment variables. The
command-line `--services` option takes precedence over the `SERVICES`
environment variable. `$REPO_ROOT` denotes the repository root.

| Category | Parameter | Default | Description |
|---|---|---|---|
| Service selection | `--services LIST` | None | Comma-separated services: `openpi`, `vggt`, `sam3`, `cosmos`, or `all`; aliases `vla` and `wam` are also accepted |
| Service selection | `SERVICES` | `openpi,vggt,sam3` | Services started when `--services` is omitted |
| Runtime | `CONDA_BIN` | `conda` | Conda executable |
| Runtime | `OPENPI_ENV` | `pi05_server` | Conda environment for OpenPI |
| Runtime | `OPENPI_PYTHON` | Empty | Direct OpenPI Python executable; bypasses `OPENPI_ENV` when set |
| Runtime | `PERCEPTION_ENV` | `EmergePolicy` | Shared Conda environment for VGGT and SAM3 |
| Runtime | `WAM_ENV` | `cosmos-policy` | Conda environment for Cosmos Policy WAM |
| Runtime | `WAM_PYTHON` | Empty | Direct WAM Python executable; bypasses `WAM_ENV` when set |
| Runtime | `PORT_CHECK_PYTHON` | Auto-detected | Python used for preflight port checks; defaults to an available `python3` or `python` |
| Port | `OPENPI_PORT` | `8000` | OpenPI service port |
| Port | `VGGT_PORT` | `8001` | VGGT service port |
| Port | `SAM3_PORT` | `8002` | SAM3 service port |
| Port | `WAM_PORT` | `8003` | Cosmos Policy WAM service port |
| GPU | `OPENPI_GPU` | `0` | OpenPI `CUDA_VISIBLE_DEVICES`; accepts comma-separated GPU IDs |
| GPU | `VGGT_GPU` | `1` | VGGT `CUDA_VISIBLE_DEVICES` |
| GPU | `SAM3_GPU` | `2` | SAM3 `CUDA_VISIBLE_DEVICES` |
| GPU | `WAM_GPU` | `0` | Cosmos Policy WAM `CUDA_VISIBLE_DEVICES` |
| OpenPI | `OPENPI_CONFIG` | `pi05_libero` | OpenPI training configuration name |
| OpenPI | `OPENPI_CHECKPOINT` | `$REPO_ROOT/checkpoints/pi05_libero` | OpenPI checkpoint directory |
| Batch | `OPENPI_MAX_BATCH_SIZE` | `4` | Maximum OpenPI requests in one inference call |
| Batch | `OPENPI_BATCH_WAIT_MS` | `10` | Maximum OpenPI batch collection time in milliseconds |
| Batch | `VGGT_MAX_BATCH_SIZE` | `4` | Maximum VGGT requests in one inference call |
| Batch | `VGGT_BATCH_WAIT_MS` | `0` | Maximum VGGT batch collection time in milliseconds |
| Batch | `SAM3_MAX_BATCH_SIZE` | `4` | Maximum SAM3 requests in one inference call |
| Batch | `SAM3_BATCH_WAIT_MS` | `0` | Maximum SAM3 batch collection time in milliseconds |
| Model path | `VGGT_CHECKPOINT` | `$REPO_ROOT/checkpoints/vggt/model.pt` | VGGT model file |
| Model path | `SAM3_CHECKPOINT` | `$REPO_ROOT/checkpoints/sam3/model.pt` | SAM3 model file |
| WAM | `WAM_POLICY_CHECKPOINT` | `$REPO_ROOT/checkpoints/cosmos-policy/Cosmos-Policy-LIBERO-Predict2-2B.pt` | Cosmos Policy checkpoint |
| WAM | `WAM_BASE_MODEL_DIR` | `$REPO_ROOT/checkpoints/cosmos-policy/Cosmos-Predict2-2B-Video2World` | Cosmos Predict2 base-model directory |
| WAM | `WAM_DATASET_STATS` | `$REPO_ROOT/checkpoints/cosmos-policy/libero_dataset_statistics.json` | LIBERO dataset statistics file |
| WAM | `WAM_T5_EMBEDDINGS` | `$REPO_ROOT/checkpoints/cosmos-policy/libero_t5_embeddings.pkl` | Pre-generated T5 embedding cache |
| WAM | `WAM_CONFIG_FILE` | `cosmos_policy/config/config.py` | WAM configuration; relative paths resolve under `WAM_PYTHONPATH` |
| WAM | `WAM_PYTHONPATH` | `$REPO_ROOT/third_party/cosmos-policy` | Cosmos Policy source directory, also added to `PYTHONPATH` |
| WAM | `WAM_EMBEDDING_MODE` | `cache-only` | T5 embedding mode, such as `cache-only` or `cache-then-online` |
| WAM | `WAM_GENERATED_T5_CACHE` | `$REPO_ROOT/artifacts/t5_embeddings/generated.pkl` | Writable cache for embeddings generated online |
| T5 | `T5_MODEL_NAME_OR_PATH` | `google-t5/t5-11b` | T5 model used for online embedding generation |
| T5 | `T5_MODEL_REVISION` | `main` | T5 model revision; pin a concrete revision for reproducibility |
| T5 | `T5_CACHE_DIR` | `$HF_HOME` or empty | Hugging Face/T5 cache directory |
| JAX | `XLA_PYTHON_CLIENT_PREALLOCATE` | `false` | Whether JAX may preallocate GPU memory |
