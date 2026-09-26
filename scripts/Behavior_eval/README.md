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
| `workspace_skill/SKILL.md` | Episode-local `vla` instructions required by the frozen π0.5 prompt. |
| `robot/drivers/behavior_r1pro_driver.py` | Native EMERGE driver consumed by `robot.controller`. |
| `robot/behavior/` | BEHAVIOR websocket protocol, status contract, and observation publishing. |
| `robot/profiles/behavior_r1pro.md` | Controller-installed robot profile. |
| `dev/behavior_agent_eval.json` | Checked-in base driver configuration. |

BEHAVIOR is no longer an external EMERGE plugin. The native driver is
registered as `behavior_r1pro` in `robot/drivers/__init__.py`; no plugin home or
runtime registration step is needed.

## Process boundary

One episode intentionally spans separate Python environments:

| Process | Environment | Role |
|---|---|---|
| Runner, Agent, Controller | `phyagentos` | EMERGE reasoning, action queue, and native driver |
| OpenPI server | `/data/yxy/openpi/.venv` | π0.5 inference |
| BEHAVIOR evaluator | `/data/yxy/envs/behavior` | Isaac Sim, episode lifecycle, and official scoring |

The evaluator only calls `env.step` while the driver grants an action budget.
While the LLM is reasoning and no action is queued, simulator time is frozen.

## Run one instance

```bash
cd /data/yxy/EMERGE-Humanoid

B1K_RUN_DIR="/data/yxy/emerge_pi05_eval/turning_on_radio_agent_$(date +%Y%m%d_%H%M%S)"

/home/embodied/miniconda3/envs/phyagentos/bin/python \
  scripts/Behavior_eval/eval_behavior_agent.py \
  --driver-config dev/behavior_agent_eval.json \
  --instance-index 0 \
  --max-steps 6000 \
  --agent-model gpt-5.6-sol \
  --agent-max-iterations 80 \
  --max-agent-feedback-rounds 2 \
  --episode-timeout 7200 \
  --write-video \
  --no-headless \
  --show-agent-events \
  --output-dir "$B1K_RUN_DIR"
```

The Agent uses EMERGE's native `~/.Emerge/config.json` provider configuration.
Pass `--config /path/to/config.json` only when an episode needs a different
configuration.

`--config` has the same meaning as the native `Emerge agent --config` option;
`--agent-config` remains an alias for older evaluation commands. The base
driver config is copied into the output directory as `driver_config.json`, with
that episode's workspace, ports, status path, and action budget applied.

`--no-headless` opens the Isaac Sim window. Omit it to run without a window.
`--show-agent-events` prints the Agent's visible decision text, tool calls, and
action state changes in the runner terminal. Provider-hidden reasoning is not
available, and is not written by EMERGE.

The Agent chooses every action through `execute_robot_action`; the runner does
not enqueue actions on its behalf. The official result remains authoritative:
`step_completed` only means that a requested action budget was consumed.

Logs are written under `<output-dir>/logs/`. The durable episode summary is
`<output-dir>/agent_episode_result.json`; official BEHAVIOR JSON and videos are
written by the evaluator in the same output tree.

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
  --output-dir "$B1K_RUN_DIR" \
  --require-success \
  --require-agent-result
```

## External BEHAVIOR requirements

The local BEHAVIOR checkout must provide:

- `OmniGibson/omnigibson/eval/eval.py` with `--status-file` support and
  `behavior.eval_status.v1` status updates.
- `OmniGibson/omnigibson/eval/r1pro_openpi.yaml`.

The runner removes inherited `ISAAC_PATH`, `ISAACSIM_PATH`, `EXP_PATH`, and
`CARB_APP_PATH` only for the evaluator subprocess, allowing the Isaac Sim
installation in `/data/yxy/envs/behavior` to configure its own paths.

## Current limits

- Only `turning_on_radio`, R1Pro, and the official single-task π0.5 checkpoint
  are wired into this entrypoint.
- Camera calibration in the observation manifest is approximate, so the Agent
  should use these views for qualitative inspection only.
- The BEHAVIOR evaluator and its R1Pro configuration remain external to this
  repository.
