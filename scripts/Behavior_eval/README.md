# BEHAVIOR-1K evaluation

This directory is the BEHAVIOR evaluation entrypoint for EMERGE. It follows the
same repository boundary as `scripts/Libero_eval`: the script owns episode
setup and process supervision, while the robot driver and transport live under
`robot/`.

The first supported setup is one `turning_on_radio` instance with the R1Pro
robot and the official single-task π0.5 checkpoint.

## Code ownership

| Path | Responsibility |
|---|---|
| `eval_behavior_agent.py` | Creates one episode workspace and supervises the Agent, Controller, OpenPI server, and BEHAVIOR evaluator. |
| `runner_common.py` | Shared process, port, workspace, and artifact helpers. |
| `robot/drivers/behavior_r1pro_driver.py` | Native EMERGE driver consumed by `robot.controller`. |
| `robot/behavior/` | BEHAVIOR websocket protocol, status contract, and observation publishing. |
| `robot/profiles/behavior_r1pro.md` | Controller-installed robot profile. |
| `dev/behavior_agent_eval.json` | Checked-in base driver configuration. |
| `dev/behavior_r1pro_openpi.yaml` | Reproducible v3.9.3 R1Pro evaluator config using the observation-key name expected by the public π0.5 checkpoint. |

BEHAVIOR is no longer an external EMERGE plugin. The native driver is
registered as `behavior_r1pro` in `robot/drivers/__init__.py`; no plugin home or
runtime registration step is needed.

## Process boundary

One episode intentionally spans separate Python environments:

| Process | Environment | Role |
|---|---|---|
| OpenPI server | `/data/yxy/openpi/.venv` | External π0.5 inference service |
| SAM3 server | `EmergePolicy` | External prompt-guided segmentation service |
| Runner, Agent, Controller | `phyagentos` | EMERGE reasoning, action queue, and native driver |
| BEHAVIOR evaluator | `/data/yxy/envs/behavior` | Isaac Sim, episode lifecycle, and official scoring |

The evaluator only calls `env.step` while the driver grants an action budget.
While the LLM is reasoning and no action is queued, simulator time is frozen.

The BEHAVIOR pipeline reserves distinct local ports so perception services can
run alongside the policy and evaluator:

| Service | Default port |
|---|---:|
| EMERGE gateway | 8000 |
| VGGT | 8001 |
| SAM3 | 8002 |
| OpenPI π0.5 | 8011 |

`--gateway-port` and `--upstream-port` may override the gateway and OpenPI
ports. They must remain different; when VGGT and SAM3 use their standard
configuration, do not assign OpenPI to 8001 or 8002.

## Run one instance

The runtime follows the original EMERGE two-terminal boundary. Terminal 1 owns
the persistent heavy-model services. Terminal 2 owns EMERGE, its Controller,
and the BEHAVIOR evaluator. Starting the SAM3 service does not make the Agent
segment every frame: only an `object_location` sub-agent call sends current
images and prompts to that service.

The original repository layout expects the SAM3 source submodule and checkpoint
at these locations:

```text
third_party/sam3/
checkpoints/sam3/model.pt
```

Initialize the submodule and install the project as described in the root
README. Then start both BEHAVIOR model services in terminal 1:

```bash
cd /data/yxy/EMERGE-Humanoid

OPENPI_GPU=0 \
SAM3_GPU=0 \
  bash scripts/Behavior_eval/start_behavior_services.sh
```

This wrapper uses the shared native
`scripts/model_server/start_external_model_servers.sh` launcher. It selects the
BEHAVIOR-specific OpenPI server on port 8011 and the standard EMERGE SAM3 server
on port 8002. VGGT is omitted because BEHAVIOR already publishes calibrated
metric depth.

In terminal 2, run the local preflight before allocating Isaac Sim or spending
API tokens:

```bash
cd /data/yxy/EMERGE-Humanoid

/home/embodied/miniconda3/envs/phyagentos/bin/python \
  scripts/Behavior_eval/eval_behavior_agent.py \
  --driver-config dev/behavior_agent_eval.json \
  --agent-model gpt-5.6-sol \
  --preflight-only
```

`--preflight-only` does not create an episode, start Isaac Sim, or call the
LLM. It requires both external model services to be healthy, validates the
provider and local BEHAVIOR files, and confirms that the EMERGE gateway port is
free.

Then start the real episode with a fresh output directory:

```bash
cd /data/yxy/EMERGE-Humanoid

/home/embodied/miniconda3/envs/phyagentos/bin/python \
  scripts/Behavior_eval/eval_behavior_agent.py \
  --driver-config dev/behavior_agent_eval.json \
  --instance-index 0 \
  --agent-model gpt-5.6-sol \
  --max-agent-feedback-rounds 2 \
  --episode-timeout 7200 \
  --write-video \
  --no-headless \
  --show-agent-events
```

When `--output-dir` is omitted, every invocation automatically uses a fresh
`/data/yxy/emerge_pi05_eval/turning_on_radio_agent_YYYYMMDD_HHMMSS/`
directory. Pass `--output-dir` only when an explicit location is required.

The Agent uses EMERGE's native `~/.Emerge/config.json` provider configuration.
Pass `--config /path/to/config.json` only when an episode needs a different
configuration.

Feedback continuation follows the native LIBERO runner: the same session is
continued only after a clean Agent response while the official evaluator still
reports an unfinished episode. Runtime failures such as `max_iterations`,
context `length`, provider errors, or timeouts are not converted into a fresh
Agent run.

`--config` has the same meaning as the native `Emerge agent --config` option;
`--agent-config` remains an alias for older evaluation commands. The base
driver config is copied into the output directory as `driver_config.json`, with
that episode's workspace, ports, status path, and action budget applied.

`--no-headless` opens the Isaac Sim window. Omit it to run without a window.
`--show-agent-events` prints the Agent's visible decision text, tool calls, and
action state changes in the runner terminal. Provider-hidden reasoning is not
available, and is not written by EMERGE.

The Agent chooses every action through `execute_robot_action`; the runner does
not enqueue actions on its behalf or install a task-specific skill. It uses the
same generic planning, object-location, attach, VLA, and verification skills as
the LIBERO path. It can combine bounded π0.5 chunks with
calibrated `object_location`, native R1Pro `move_to_pose` / `move_linear` /
`set_gripper` actions, and `task_verification`. The official result remains authoritative:
`step_completed` only means that a requested action budget was consumed. During
the rollout, EMERGE maintains the live episode state from its controlled
websocket gateway. During a run, `evaluation.steps` is null because the stock
evaluator does not publish its official counter until completion;
`runtime_progress.gateway_action_steps` is a clearly labelled gateway-side
diagnostic. At completion the runner reads the result JSON written by the stock
BEHAVIOR evaluator; that official JSON remains the source of success and score.

Logs are written under `<output-dir>/logs/`. The durable episode summary is
`<output-dir>/agent_episode_result.json`; official BEHAVIOR JSON and videos are
written by the evaluator in the same output tree.

The runner connects to the already-running OpenPI and SAM3 endpoints; it does
not own either model process. It refuses to start Isaac Sim when either service
is unhealthy, preventing an expensive run from silently dropping π0.5 control
or measured localization. BEHAVIOR uses native calibrated RGB-D, so an Agent
config that explicitly selects `geometryBackend: vggt` is rejected.

## Verify

The offline suite exercises the driver, websocket gate, action queue,
observation snapshots, runner lifecycle, and artifact verification without
starting Isaac Sim or calling an LLM:

```bash
/home/embodied/miniconda3/envs/phyagentos/bin/python \
  -m unittest discover -s scripts/Behavior_eval/checks -p 'test_*.py' -v
```

Verify a finished episode using only its artifacts:

```bash
/home/embodied/miniconda3/envs/phyagentos/bin/python -B \
  -m scripts.Behavior_eval.checks.verify_artifacts \
  --output-dir /data/yxy/emerge_pi05_eval/turning_on_radio_agent_YYYYMMDD_HHMMSS \
  --require-success \
  --require-agent-result
```

Replace `YYYYMMDD_HHMMSS` with the directory name printed by the completed
runner.

## External BEHAVIOR requirements

The local BEHAVIOR checkout must be the unmodified official `v3.9.3` release and
provide:

- `OmniGibson/omnigibson/eval/eval.py` with the `BatchedEvaluator` entrypoint.
- the standard R1Pro assets and controllers used by the bundled evaluator.

The runner uses `--num-envs 1` and disables optional client-side action-chunk
replay so that each gateway permit still corresponds to one simulator action.
It selects BEHAVIOR's stock `RGBDFullResWrapper`: π0.5 continues to consume the
three RGB keys and resizes them internally to 224x224, while EMERGE saves the
aligned metre-valued `depth_linear` arrays as a read-only side copy. The raw
websocket message forwarded to OpenPI is unchanged.
It does not require patches or additional files in the BEHAVIOR repository.
The checked-in `dev/behavior_r1pro_openpi.yaml` copies the official v3.9.3 R1Pro
settings but names the runtime robot `robot`, because the public OpenPI wrapper
was trained against `robot::proprio` and `robot::robot:...` observation keys.

The runner removes inherited `ISAAC_PATH`, `ISAACSIM_PATH`, `EXP_PATH`, and
`CARB_APP_PATH` only for the evaluator subprocess, allowing the Isaac Sim
installation in `/data/yxy/envs/behavior` to configure its own paths.

## Perception services for measured approach

SAM3 remains an external persistent service exactly as in the original EMERGE
LIBERO path. The service loads the model once and waits on port 8002. The Main
Agent decides when to delegate to `object_location`; its specialist inspects
the current views, constructs appearance prompts, invokes `segment_candidates`,
reviews the returned overlays, and only then requests three-dimensional
localization. `locate_candidates` reuses the cached masks. BEHAVIOR replaces
the original VGGT geometry stage with its calibrated `depth_linear` arrays;
the Agent/SAM3 delegation contract remains unchanged.

## Current limits

- Only `turning_on_radio`, R1Pro, and the official single-task π0.5 checkpoint
  are wired into this entrypoint.
- Camera RGB, linear depth, extrinsics, and current R1Pro proprioception are
  copied from the official evaluator observation into one revision-consistent
  manifest without changing the OpenPI wire message. Extrinsics and
  localization results use the `robot_base` coordinate frame; checked-in
  intrinsics match the official v3.9.3 full-resolution RGB-D wrapper.
- The BEHAVIOR evaluator and its R1Pro configuration remain external to this
  repository.
