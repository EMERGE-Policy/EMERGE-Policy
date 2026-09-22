# External Model Servers

Run from the repository root. Complete the [installation](../../README.md#installation)
and [checkpoint setup](../../README.md#model-checkpoints) first.

## 1. Start services

Choose one stack. Stop it with `Ctrl+C`.

### WAM: Cosmos + VGGT + SAM3 (default)

```bash
WAM_GPU=0 \
VGGT_GPU=1 \
SAM3_GPU=2 \
bash scripts/model_server/start_external_model_servers.sh --services cosmos,vggt,sam3
```

### VLA: OpenPI + VGGT + SAM3

```bash
OPENPI_GPU=0 \
VGGT_GPU=1 \
SAM3_GPU=2 \
bash scripts/model_server/start_external_model_servers.sh --services openpi,vggt,sam3
```

Without `--services`, the script starts `cosmos,vggt,sam3`.
WAM uses the first GPU listed in `WAM_GPU`.

## 2. Check services

Run in another terminal:

```bash
# WAM
curl --noproxy '*' http://127.0.0.1:8003/healthz

# VLA
curl --noproxy '*' http://127.0.0.1:8000/healthz

# Shared perception services
curl --noproxy '*' http://127.0.0.1:8001/healthz
curl --noproxy '*' http://127.0.0.1:8002/healthz
```

Public endpoints return JSON, including `schema: emerge.model-health.v1`,
`service`, `instance_id`, `model_id`, `input_schema`, `output_schema`,
`capabilities`, `protocol_version`, and `status`.
HTTP 200 means `ready`; HTTP 503 reports `starting`, `draining`, or `failed`.
Loading happens after the public listener starts, so startup is observable.
Plain `OK` is not a valid public health response.

The interactive CLI's `/health` discovers services on `127.0.0.1:8000–8099`
and checks the local port range. Names come from the responding services; gaps
in the port range don't stop discovery. See
[client discovery configuration](../../Emerge/README.md#external-model-services).

| Service | Conda environment | Port |
| --- | --- | ---: |
| Cosmos WAM | `cosmos-policy` | 8003 |
| OpenPI VLA | `pi05_server` | 8000 |
| VGGT | `EmergePolicy` | 8001 |
| SAM3 | `EmergePolicy` | 8002 |

OpenPI uses the shared runtime directly. `openpi_server` loads the official
OpenPI policy in its adapter, while `openpi_policy.py` supplies the
policy-specific transforms and native batch inference helpers. There is one
process, one endpoint, and no second network hop.

Start OpenPI directly with:

```bash
python -m external_model_server.openpi_server \
  --config-name pi05_libero \
  --checkpoint-dir checkpoints/pi05_libero \
  --port 8000
```


## 3. Run evaluation

Activate `EmergePolicy` and follow the selected guide:

| Evaluation | VLA | WAM |
| --- | --- | --- |
| [LIBERO](../Libero_eval/README.md) | `--policy-backend vla` | `--policy-backend wam` (default) |
| [LIBERO-Plus](../LiberoPlus_eval/README.md) | `--policy-backend vla` | `--policy-backend wam` (default) |
| [LIBERO-Pro](../LiberoPro_eval/README.md) | Supported | Not supported |

## Optional: batching for multiple workers

All four services default to batch size **1** when started through the Bash script.

WAM, batch size 4:

```bash
WAM_GPU=0 \
VGGT_GPU=1 \
SAM3_GPU=2 \
WAM_MAX_BATCH_SIZE=4 \
VGGT_MAX_BATCH_SIZE=4 \
SAM3_MAX_BATCH_SIZE=4 \
VGGT_BATCH_WAIT_MS=10 \
SAM3_BATCH_WAIT_MS=10 \
bash scripts/model_server/start_external_model_servers.sh --services cosmos,vggt,sam3
```

VLA, batch size 4:

```bash
OPENPI_GPU=0 \
VGGT_GPU=1 \
SAM3_GPU=2 \
OPENPI_MAX_BATCH_SIZE=4 \
VGGT_MAX_BATCH_SIZE=4 \
SAM3_MAX_BATCH_SIZE=4 \
VGGT_BATCH_WAIT_MS=10 \
SAM3_BATCH_WAIT_MS=10 \
bash scripts/model_server/start_external_model_servers.sh --services openpi,vggt,sam3
```

Use `--workers 4` in the evaluator. WAM candidate count is set separately in
`dev/libero_agent_eval.json` or `dev/libero_plus_eval.json`:
`wam.search.num_candidates`.
WAM batching supports official cached task text with `none` or `joint_value`
scoring; other configurations use single-request inference.

| Service | Batch size variable | Default | Wait variable | Default (ms) |
| --- | --- | ---: | --- | ---: |
| WAM | `WAM_MAX_BATCH_SIZE` | 1 | `WAM_BATCH_WAIT_MS` | 10 |
| VLA | `OPENPI_MAX_BATCH_SIZE` | 1 | `OPENPI_BATCH_WAIT_MS` | 10 |
| VGGT | `VGGT_MAX_BATCH_SIZE` | 1 | `VGGT_BATCH_WAIT_MS` | 0 |
| SAM3 | `SAM3_MAX_BATCH_SIZE` | 1 | `SAM3_BATCH_WAIT_MS` | 0 |

## Optional: WAM phase conditioning

Task conditioning with the official T5 cache is the default. To use `phase`
or `task_with_phase`, start WAM with online T5 encoding:

```bash
WAM_GPU=0 \
VGGT_GPU=1 \
SAM3_GPU=2 \
WAM_EMBEDDING_MODE=cache-then-online \
bash scripts/model_server/start_external_model_servers.sh --services cosmos,vggt,sam3
```

Then pass `--wam-conditioning-mode phase` or
`--wam-conditioning-mode task_with_phase` to LIBERO or LIBERO-Plus evaluation.

## Other settings

Override settings as environment variables before the launch command.

| Setting | Default |
| --- | --- |
| `OPENPI_ENV` / `WAM_ENV` / `PERCEPTION_ENV` | `pi05_server` / `cosmos-policy` / `EmergePolicy` |
| `OPENPI_GPU` / `WAM_GPU` / `VGGT_GPU` / `SAM3_GPU` | `0` / `0` / `1` / `2` |
| `OPENPI_PORT` / `WAM_PORT` / `VGGT_PORT` / `SAM3_PORT` | `8000` / `8003` / `8001` / `8002` |
| `MODEL_QUEUE_CAPACITY` / `MODEL_SHUTDOWN_TIMEOUT` | `64` / `30` seconds |
| `OPENPI_CHECKPOINT` | `checkpoints/pi05_libero` |
| `VGGT_CHECKPOINT` / `SAM3_CHECKPOINT` | `checkpoints/vggt/model.pt` / `checkpoints/sam3/model.pt` |
| `WAM_POLICY_CHECKPOINT` | `checkpoints/cosmos-policy/Cosmos-Policy-LIBERO-Predict2-2B.pt` |
| `WAM_BASE_MODEL_DIR` | `checkpoints/cosmos-policy/Cosmos-Predict2-2B-Video2World` |
| `WAM_DATASET_STATS` | `checkpoints/cosmos-policy/libero_dataset_statistics.json` |
| `WAM_T5_EMBEDDINGS` | `checkpoints/cosmos-policy/libero_t5_embeddings.pkl` |

Default checkpoint paths are relative to the repository root.
Set `OPENPI_PYTHON` or `WAM_PYTHON` to an absolute Python executable to override
the corresponding Conda environment.

Full parameter list:

```bash
bash scripts/model_server/start_external_model_servers.sh --help
```
