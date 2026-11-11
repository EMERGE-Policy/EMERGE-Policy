# Configure and run AC-WM

## Environments

Install the small adapter dependencies into the existing Emerge environment:

```bash
python -m pip install -e '.[ac-wm]'
git submodule update --init third_party/cosmos-framework
```

The Cosmos Framework submodule is pinned to
`cf5d68c00d97ccd2480a2320ed652b92dec63102`. Install its CUDA dependencies in a
separate environment using the pinned [upstream setup guide](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/docs/setup.md).
Cosmos3 and Cosmos Policy WAM have separate model environments and services.

Set these variables to the installed Cosmos3 environment and model assets:

| Variable | Value |
| --- | --- |
| `COSMOS_PYTHON` | Absolute path to the Cosmos Framework environment's Python executable |
| `COSMOS_CHECKPOINT` | Cosmos3-Edge checkpoint directory or supported checkpoint identifier |
| `COSMOS_CONFIG` | Absolute path to its Cosmos Framework inference configuration |
| `COSMOS_LIBERO_ACTION_STATS` | Pinned `libero_native_frame_wise_relative_rot6d.json` file, set below |
| `COSMOS_ACTION_HORIZON` | Maximum real action rows per prediction, a multiple of four; default `52` |

From the EMERGE repository root:

```bash
export COSMOS_LIBERO_ACTION_STATS="$PWD/third_party/cosmos-framework/cosmos_framework/data/generator/action/normalizer_stats/libero_native_frame_wise_relative_rot6d.json"
export COSMOS_ACTION_HORIZON=52
```

## Start the Cosmos3 service

Run this command from the EMERGE repository root using the Cosmos environment.
The environment needs `websockets>=16,<17`, `msgpack` and `numpy` for EMERGE's
shared model-service transport, in addition to Cosmos Framework itself.

```bash
mkdir -p tmp/ac-wm-service
export TMPDIR="$PWD/tmp/ac-wm-service" TMP="$PWD/tmp/ac-wm-service" TEMP="$PWD/tmp/ac-wm-service"
"$COSMOS_PYTHON" -m external_model_server.cosmos3_server \
  --checkpoint "$COSMOS_CHECKPOINT" \
  --config-file "$COSMOS_CONFIG" \
  --output-dir "$PWD/tmp/ac-wm-service" \
  --host 127.0.0.1 --port 8773
```

The pipeline loads once and serves candidate batches through the same
`ModelServerRuntime` protocol as the other EMERGE services. Its service name is
`cosmos3_ac_wm`; the default port is `8773`, separate from WAM's `8003`.
Check readiness with `curl --noproxy '*' http://127.0.0.1:8773/healthz`.

In the terminal that starts Emerge:

```bash
export COSMOS_SERVER_URL=ws://127.0.0.1:8773
```

Agent and service processes share the same filesystem: the request contains
absolute input-spec paths and the response contains generated-video paths.
If `COSMOS_SERVER_URL` is unset, the adapter instead invokes the Cosmos
Framework CLI through `COSMOS_PYTHON` for each decision. `COSMOS_GPU` controls
the CLI subprocess's visible GPU; for a resident service, set
`CUDA_VISIBLE_DEVICES` when starting that service.

## Enable the module and select a judge

Add this section to the active Emerge configuration. The named provider must
already have an OpenAI-compatible `apiBase` and `apiKey` configured:

```json
{
  "acWm": {
    "enabled": true,
    "judgeModel": "gpt-6-astra",
    "judgeProvider": "custom",
    "dispatchMode": "evaluated_prefix",
    "onNoPositive": "reject"
  }
}
```

`ac_wm` and snake_case field names are also accepted. The judge model is
independent of the main planning model. `EMERGE_AC_WM=1` is an alternative to
`enabled: true`; without either setting the module is disabled.

Start the normal controller and Emerge entry point with these variables in
the process environment. AC-WM does not change `EMERGE_POLICY_BACKEND` or the
WAM search configuration.

## Request a recovery comparison

Example arguments for `execute_robot_action` after a failed grasp, using a
pre-grasp pose measured from the current scene:

```json
{
  "action_type": "ac_wm_select",
  "reasoning": "Compare retrying the grasp with re-anchoring at the measured pre-grasp pose.",
  "parameters": {
    "plan_context": {"current_subgoal": "recover the grasp of the black bowl"},
    "candidates": [
      {
        "candidate_id": "vla_retry",
        "skill": "vla_execute",
        "planner_priority": 1,
        "parameters": {"instruction": "pick up the black bowl next to the plate", "step": 100}
      },
      {
        "candidate_id": "reanchor",
        "skill": "move_to_pose",
        "planner_priority": 2,
        "parameters": {"position_m": [0.05, 0.12, 1.02], "orientation_euler": [3.14, 0.0, 0.0]}
      }
    ]
  }
}
```

Replace the example pose with the measured pose. Each decision writes
`proposal-summary.json`, `decision.json`, `events.jsonl`, input action arrays,
prediction videos and judge frames under the active workspace's
`artifacts/ac-wm/<decision-id>/`. The decision records distinguish forecast
selection, planner-priority tie breaks, fallback and dispatch outcome.

## Regression tests

The CPU tests cover non-executing skill proposals, shared observations,
selection and dispatch modes, Cosmos action conversion, decoded judge frames,
the actual WebSocket service protocol, and ordinary VLA/WAM execution routing.
Model outputs are supplied by test doubles.

```bash
python -m pip install pytest
mkdir -p tmp/ac-wm-tests
export TMPDIR="$PWD/tmp/ac-wm-tests" TMP="$PWD/tmp/ac-wm-tests" TEMP="$PWD/tmp/ac-wm-tests"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/ac_wm \
  --basetemp="$PWD/tmp/ac-wm-tests/pytest" \
  -o cache_dir="$PWD/tmp/ac-wm-tests/cache"
```

For model evaluation, record the decision artifacts together with the
environment's task-success result. The VLM utility score describes predicted
local progress; a dispatched skill or positive score alone does not establish
task completion.
