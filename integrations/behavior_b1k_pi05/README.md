# BEHAVIOR-1K π0.5 gateway

This integration connects EMERGE to the BEHAVIOR-1K `turning_on_radio` challenge.
The official BEHAVIOR evaluator still owns OmniGibson, episode reset, termination
and scoring; the gateway relays its websocket frames unchanged to the official
OpenPI B1K server, and the driver bounds how many policy actions may pass
through.

> **Status.** The transport, the action queue, the driver contract and the
> acceptance checker are covered by tests that run without a GPU. A successful
> official episode has **not** been reproduced by this branch; see
> [Acceptance](#acceptance) for what a pass actually requires.

## Layout

| File | Role |
|---|---|
| `contracts.py` | Frozen instruction, driver-config validation, evaluator-status validation and episode verdicts. Standard library only. |
| `gateway.py` | Websocket relay plus the execution gate. Owns no simulator. |
| `driver.py` | EMERGE `BaseDriver` exposing bounded `vla_execute`. |
| `observation.py` | Decodes a side copy of the camera frames into the workspace. |
| `runner_common.py` | Process-group supervision, port ownership, workspace preparation. |
| `workspace_skill/SKILL.md` | The `vla` skill installed into the episode workspace. |
| `run_turning_on_radio.py` | Policy-transparency smoke: evaluator → gateway → π0.5, no Agent. |
| `run_action_queue_smoke.py` | The same episode driven through the real Controller and `ACTION.md`. |
| `run_turning_on_radio_agent.py` | Full episode driven by the real EMERGE Agent. |
| `checks/` | Unit and integration tests plus `verify_artifacts.py`. |

## Processes and Python environments

An episode spans four processes in three environments. They must not be merged.

| Process | Interpreter | Owns |
|---|---|---|
| Runner | any (EMERGE deps for the Agent path) | supervision, workspace, plugin registry |
| Agent + Controller | `phyagentos` | `ACTION.md`, `ROBOT_STATE.md`, the gateway (daemon thread in the Controller) |
| OpenPI π0.5 server | `/data/yxy/openpi/.venv` | policy inference |
| BEHAVIOR evaluator | `/data/yxy/envs/behavior` | **the only simulator owner** |

## 1. Policy-transparency smoke run

```bash
cd /data/yxy/EMERGE-Humanoid
/data/yxy/openpi/.venv/bin/python \
  integrations/behavior_b1k_pi05/run_turning_on_radio.py \
  --instance-index 0 --max-steps 500 --no-write-video
```

Starts the π0.5 server on 8001, the standalone gateway on 8000, and the
evaluator. Each run gets a fresh timestamped output directory under
`/data/yxy/emerge_pi05_eval/`; reusing one is refused. Logs and the official
result JSON live there.

Ports are claimed by binding before any process starts, so a leftover service is
reported instead of being mistaken for this run's.

The runner removes inherited `ISAAC_PATH`, `ISAACSIM_PATH`, `EXP_PATH`, and
`CARB_APP_PATH` values only for the evaluator subprocess. This lets the Isaac
Sim package installed in `/data/yxy/envs/behavior` configure its matching paths
and prevents a stale shell-level Isaac Sim directory from affecting the run.

> Passing this smoke shows that frames were relayed without an error. It does
> **not** prove numerical policy equivalence; that needs a direct-connection
> comparison over a fixed observation sequence.

## 2. Gateway only

```bash
/data/yxy/openpi/.venv/bin/python \
  integrations/behavior_b1k_pi05/gateway.py \
  --upstream-url ws://127.0.0.1:8001 --listen-host 127.0.0.1 --listen-port 8000
```

Do not run this alongside the Controller path: both would try to own port 8000.

## 3. ACTION.md controlled smoke run

```bash
cd /data/yxy/EMERGE-Humanoid
/home/embodied/miniconda3/envs/phyagentos/bin/python \
  integrations/behavior_b1k_pi05/run_action_queue_smoke.py \
  --instance-index 0 --max-steps 500 --no-write-video
```

This starts the gateway inside `robot.controller` as a registered external
driver, waits until the evaluator is connected, the first status frame and the
first camera frame have arrived, then places bounded `vla_execute` actions in
the real action queue.

To prove that the simulator only advances while an action is queued, split the
budget into chunks:

```bash
/home/embodied/miniconda3/envs/phyagentos/bin/python \
  integrations/behavior_b1k_pi05/run_action_queue_smoke.py \
  --instance-index 0 --max-steps 20 --chunk-steps 5 --chunk-pause 2
```

Between chunks the runner asserts that the evaluator's step counter did not
move. The workspace keeps `ACTION.md`, `ROBOT_STATE.md`, the installed
`EMBODIED.md` profile, `BEHAVIOR_STATUS.json` and the camera manifest.

## 4. Real EMERGE Agent run

```bash
cd /data/yxy/EMERGE-Humanoid
/home/embodied/miniconda3/envs/phyagentos/bin/python \
  integrations/behavior_b1k_pi05/run_turning_on_radio_agent.py \
  --instance-index 0 --max-steps 20000 --agent-max-iterations 80 \
  --max-agent-feedback-rounds 2 --episode-timeout 7200 \
  --write-video --require-success
```

The runner installs `workspace_skill/SKILL.md` as `<workspace>/skills/vla/SKILL.md`,
which shadows the builtin `vla` skill: the builtin demands a phase-local
instruction, and this checkpoint accepts only its exact frozen sentence. Both
rules cannot apply at once.

The Agent chooses every action through `execute_robot_action`; the runner never
queues one on its behalf. A run that stops early is resumed **only** when the
runtime reported a benign reason (`max_iterations`, `length`), no action is still
pending or running, and feedback rounds remain. `cancellation_unconfirmed`,
`unfinished_action`, `cleanup_error` and similar reasons stop the episode
instead, because retrying work that is not known to have stopped is unsafe.

Exit codes:

| Situation | Exit |
|---|---|
| Official success | 0 |
| Episode terminated normally, `success=false`, without `--require-success` | 0 |
| Episode terminated normally, `success=false`, with `--require-success` | 1 |
| `phase == "error"`, timeout, or any infrastructure failure | 1 |

The result is always written to `agent_episode_result.json`, including on every
failure path, together with `verdict` and `infrastructure_error`.

`--agent-model` is optional when `~/.Emerge/config.json` already has a usable
provider. The provider preflight runs in the runner's interpreter; when that
differs from `--agent-python`, the runner warns that the check is not conclusive
for the Agent.

## 5. Automated acceptance

```bash
/home/embodied/miniconda3/envs/phyagentos/bin/python -B \
  -m integrations.behavior_b1k_pi05.checks.verify_artifacts \
  --output-dir <episode output dir> --require-success --require-agent-result
```

Prints one line per check and `PASS`, or the failing checks and a non-zero exit.
It reads only artifacts, so it needs no GPU.

<a id="acceptance"></a>
## Acceptance

Six levels, cheapest first. Only the last one is success.

| Level | Command | Needs | Passes when |
|---|---|---|---|
| Unit | `python -m unittest discover -s integrations/behavior_b1k_pi05/checks -p 'test_*.py'` | nothing | all tests pass |
| Queue integration | `test_action_queue` | nothing | real `_poll_once` drives the real queue and gateway to a synchronized result |
| Transport smoke | `run_turning_on_radio.py` | GPU + Isaac Sim | evaluator reaches its result JSON and all processes exit |
| Queue smoke | `run_action_queue_smoke.py` | GPU + Isaac Sim | the queued action completes and the evaluator does not step while idle |
| Agent end-to-end | `run_turning_on_radio_agent.py` | GPU + Isaac Sim + an LLM | the Agent produces the actions, reads state after each, and never claims false success |
| **Task success** | `verify_artifacts.py --require-success` | nothing extra | `phase == "completed"`, `done == true`, `success == true`, and the official result JSON agrees |

Final success additionally requires: the result path is inside this run's output
directory, task/instance/rollout agree between the live status and the result
file, no action was left pending or running, every referenced camera image
exists at the published revision, and `agent_episode_result.json` agrees with the
official outcome.

## External dependency (not vendored here)

This integration is delivered without any BEHAVIOR-side changes. Two external
requirements must exist in the local BEHAVIOR-1K checkout and are intentionally
**not** copied into this repository:

1. `OmniGibson/omnigibson/eval/eval.py` must support `--status-file` and publish
   `behavior.eval_status.v1` documents (`task`, `instance_id`, `rollout_id`,
   `phase`, `steps`, `success`, `done`, `terminated`, `truncated`,
   `termination_reason`, `result_path`, `q_score`, `error`). This is a local,
   uncommitted modification.
2. `OmniGibson/omnigibson/eval/r1pro_openpi.yaml` must exist. It is an untracked
   local file.

Without both, the action-queue and Agent paths cannot start. Re-verify them when
the BEHAVIOR checkout moves.

## Known limitations

- The provider preflight is conclusive only when the runner and `--agent-python`
  interpreters match.
- Camera intrinsics are approximations and `T_world_camera` is a placeholder
  identity, so the views must not be used for metric geometry.
- The visual monitor is not disabled by these runners. If
  `visual_monitor.enabled` is on, an image-based completion judgement can cancel
  an action before the official evaluator agrees, so disable it in the run
  config when validating acceptance. Manual camera review stays available.
- A cancel cannot retract an action that was already forwarded; the gateway
  accounts for that reply so it cannot shorten the next action's budget, but the
  evaluator still applies it.
- No successful official episode has been reproduced here. If a long-budget
  direct run also fails, the cause is the checkpoint or the task, not this
  integration's plumbing.
