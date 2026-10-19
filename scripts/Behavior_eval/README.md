# BEHAVIOR-1K evaluation

This directory contains EMERGE's integration around an unmodified, pinned
BEHAVIOR-1K evaluator. The champion PI_BEHAVIOR service performs policy
inference; BEHAVIOR owns simulation and official scoring; EMERGE can be added
first as a read-only shadow gateway and later as the active controller.

## Repository and process boundaries

| Component | Location / environment | Responsibility |
|---|---|---|
| Official evaluator | `third_party/behavior-1k`, `/data/yxy/envs/behavior` | Isaac Sim, task lifecycle, official JSON and video |
| Champion policy | `third_party/behavior-1k-solution/.venv` | Modified OpenPI / PI_BEHAVIOR inference on port 8011 |
| EMERGE | this repository, `phyagentos` | Gateway, observations, Agent and Controller |
| SAM3 | `third_party/sam3`, `EmergePolicy` | Prompt-guided segmentation on port 8002 |
| LightNav | `third_party/lightnav-0`, `/data/yxy/envs/lightnav` | Optional navigation inference on port 8051 |

The default ports are EMERGE gateway `8000`, VGGT `8001`, SAM3 `8002`, and
champion OpenPI `8011`, and LightNav `8051`. BEHAVIOR uses calibrated RGB-D, so the normal path does
not need VGGT.

The general runner contains no task name or task instruction. Those values
come from `dev/behavior_agent_eval.json`, or from the `--task-name` and
`--task-instruction` overrides. It does not inject task-specific execution
routes or schedule skill calls. Robot-specific action and evaluator contracts
belong in `robot/profiles/behavior_r1pro.md`, which the Controller installs as
the episode's `EMBODIED.md`; the Agent chooses when to use its skills.

## Output layout

All experiments use two top-level groups:

```text
/data/yxy/outputs/
├── champion_baseline/
│   └── <task>/
│       ├── json/
│       ├── videos/
│       └── logs/
└── Emerge/
    └── <task>/
        ├── json/
        ├── videos/
        ├── logs/
        ├── workspace/
        └── shadow/
```

`shadow/` is deliberately nested below the task so a read-only integration
test cannot collide with a later active-Agent episode. Passing `--output-dir`
still overrides the active runner's default.

## Start the champion policy and SAM3

In terminal 1:

```bash
cd /data/yxy/EMERGE-Humanoid

OPENPI_GPU=0 \
SAM3_GPU=0 \
  bash scripts/Behavior_eval/start_behavior_services.sh
```

The launcher expects the champion checkpoints under
`/data/yxy/checkpoints/behavior_submission_hfmirror/checkpoint_{1,2,3,4}` and
uses `third_party/behavior-1k-solution/task_checkpoint_mapping.json`.

## Champion baseline

The baseline connects the official evaluator directly to port 8011. It runs
the official ten `public_test` indices (`0` through `9`), writes video, skips
existing result JSON files, and contains no Python loop embedded in the shell
command:

```bash
cd /data/yxy/EMERGE-Humanoid
bash scripts/Behavior_eval/run_champion_baseline.sh turning_on_radio
```

Results are written to
`/data/yxy/outputs/champion_baseline/turning_on_radio/`.

To run every task assigned to one champion checkpoint, use the checkpoint
orchestrator. It reads task IDs from the mapping JSON, resolves names from the
official challenge metadata, and invokes the single-task runner sequentially:

```bash
cd /data/yxy/EMERGE-Humanoid

# Inspect the exact task list without starting Isaac Sim.
bash scripts/Behavior_eval/run_checkpoint_baseline.sh checkpoint_2 --list-only

# Run checkpoint 2's 16 tasks x 10 official reporting instances.
bash scripts/Behavior_eval/run_checkpoint_baseline.sh checkpoint_2
```

The command is resumable: completed JSON files are skipped, a failed task is
recorded while later tasks continue, and rerunning the same command fills only
missing instances.

## Stage 3: read-only EMERGE shadow

### Current Agent vs champion: radio public 10

Run the current, unmodified Agent entrypoint sequentially on `public_test`
indices 0–9 (instance IDs 301–310), one rollout each:

```bash
cd /data/yxy/EMERGE-Humanoid
bash scripts/Behavior_eval/run_radio_agent_public10.sh
```

Keep the existing champion checkpoint-2, SAM3 and optional LightNav services
running as usual, and do not run another evaluator on gateway port 8000.
This Bash wrapper does not launch model services or change policy/profile
logic. It uses `gpt-6.1-sol`, the official default task step limit, 3600 seconds
wall time per episode, video recording, and the same GUI setting as the latest
single-instance experiment. Pass `--headless` to hide simulator windows.
Keep code, profiles, model checkpoints and service settings fixed across all ten.

New results are separate from earlier tuning runs:

```text
/data/yxy/outputs/Emerge/turning_on_radio/public10/
├── instance_301/   # json/, videos/, logs/, workspace/, agent_episode_result.json
├── instance_302/
└── ... instance_310/
```

Compare the exact same IDs with
`/data/yxy/outputs/champion_baseline/turning_on_radio/json/`. Earlier successful
debugging episodes are not substituted into this new cohort. Report official
successes out of ten, mean final Q-score, and both all-episode and success-only
steps; infrastructure failures are reported separately, not silently omitted.

Rerunning skips only identity-checked episodes with both an official result and
a clean completed episode summary. Official `success: false` is a completed
experiment and is also skipped. Incomplete/invalid existing directories are
preserved and reported without overwriting; inspect and move that instance
directory aside before retrying it. Ctrl+C stops the batch after the current
runner handles interruption; it does not schedule the next instance. End-of-run
statistics describe only completed episodes until all ten are present.
`EMERGE_RADIO_OUTPUT_DIR` can select another cohort root; `EMERGE_AGENT_PYTHON`
and `EMERGE_AGENT_MODEL` can select another Agent runtime/model, but must remain
fixed within a comparison cohort.

For the active navigation/body-observation experiment, see
[`integrations/lightnav_behavior/README.md`](../../integrations/lightnav_behavior/README.md#agent-guided-navigation-and-body-observation).
It reuses `eval_behavior_agent.py --navigation-only`. The Agent may choose
multiple VLN calls, bounded torso/arm posture changes, and visual/location
subagents. Navigation and posture share one total simulator-step budget;
the runner records every call and the full official video. This adds active
observation, not automatic collision avoidance or verified arrival.

Run one instance through the EMERGE gateway without an Agent or Controller:

```bash
cd /data/yxy/EMERGE-Humanoid
bash scripts/Behavior_eval/run_emerge_shadow.sh turning_on_radio 0
```

The second argument is a `public_test` index from `0` to `9`. The gateway is
started without `--controlled`, so metadata, observations, reset messages, and
champion actions are forwarded unchanged. A decoded side copy of the RGB-D
observation is saved under
`/data/yxy/outputs/Emerge/turning_on_radio/shadow/workspace/`; the official
JSON, video and logs stay under the same `shadow/` tree. This stage calls no
LLM and is the first end-to-end integration check.

## Active EMERGE Agent

Complete-task runs expose both the built-in `Emerge/skills/lightnav/SKILL.md`
and the existing VLA skill through the normal skill loader. The Agent chooses
when to call `navigate_vln` or `vla_execute`; the runner does not impose a
navigation-first sequence or a handoff point. Do **not** pass
`--navigation-only` for a complete task. That flag remains a separate debug
experiment that disables VLA and uses the navigation profile.

For navigation, start this service in another terminal:

```bash
/data/yxy/envs/lightnav/bin/lightnav-serve \
  --task vln --backend hf --aspect_mode keep \
  --model_path /data/yxy/checkpoints/lightnav-0 --port 8051
```

Only the champion server is mandatory for complete-task startup. Preflight
checks LightNav and SAM3 and reports unavailable optional services; their
absence does not block VLA. LightNav is mandatory for `--navigation-only`.
SAM3 is needed only if the Agent invokes object localization, not for VLA,
VLN, or ordinary visual verification. To start the champion alone, use
`BEHAVIOR_SERVICES=openpi bash scripts/Behavior_eval/start_behavior_services.sh`.

First run a preflight. It does not start Isaac Sim or call the LLM:

```bash
cd /data/yxy/EMERGE-Humanoid

/home/embodied/miniconda3/envs/phyagentos/bin/python \
  scripts/Behavior_eval/eval_behavior_agent.py \
  --driver-config dev/behavior_agent_eval.json \
  --agent-model gpt-6.1-sol \
  --preflight-only
```

Then run one active episode:

```bash
cd /data/yxy/EMERGE-Humanoid

/home/embodied/miniconda3/envs/phyagentos/bin/python \
  scripts/Behavior_eval/eval_behavior_agent.py \
  --driver-config dev/behavior_agent_eval.json \
  --instance-index 0 \
  --agent-model gpt-6.1-sol \
  --max-agent-feedback-rounds 2 \
  --episode-timeout 7200 \
  --write-video \
  --no-headless \
  --show-agent-events
```

With the checked-in config, the default output is
`/data/yxy/outputs/Emerge/turning_on_radio/`. The directory must not already
contain active-episode `json/` or `agent_episode_result.json` artifacts; this
guard prevents stale results from being mistaken for the current rollout.

Do not pass `--max-steps` unless an experiment intentionally overrides the
benchmark. When omitted, the stock evaluator uses the selected task's official
configured limit. `--no-headless` opens the Isaac Sim window; omit it for
headless execution.

The Agent uses `~/.Emerge/config.json` unless `--config` is supplied. The
runner supervises the native Controller and controlled gateway, while the
R1Pro driver enforces the champion checkpoint's fixed instruction on every
VLA call. Skill choice and timing remain with the Agent. Success comes only
from the official result JSON; a completed action chunk is not itself task
success.

There is no `navigation`/`manipulation` control-mode switch or
`set_control_mode` action. Champion observations and all 23 VLA action
dimensions are relayed unchanged, including base velocities. The checkpoint,
fixed task instruction, robot config, RGB-D wrapper, and official scoring
remain the same as the champion baseline. Consecutive VLA calls preserve the
policy's internal state and normal action cache. After VLN/local motion the
gateway sends `{"invalidate_action_cache": true}` before the next VLA
observation. This clears only old actions, their index and rolling-inpainting
seeds; task ID, current stage, voting history, counters and base-policy state
are preserved. The next action is inferred from fresh observations. Official
`{"reset": true}` still starts a new episode with the original full reset;
task changes retain their original stage-reset behavior. No model weights,
stage definitions or correction rules are changed. External movement and
fresh action planning still differ from uninterrupted baseline execution.

Restart the champion service after this update. The handshake must advertise
`emerge_capabilities.invalidate_action_cache: true`. A legacy server can
still serve plain VLA, but a local-motion handoff is rejected explicitly;
there is no fallback to resetting the task stage. Baseline clients need no
new flag and keep their normal observation/action/reset protocol.

Complete-task `agent_episode_result.json` also records `navigation_calls`,
`navigation.nav_steps`, `posture_calls`, `total_robot_action_steps`, and
`upstream_cache_invalidations_sent` and `evaluator_resets_forwarded` as separate
counts for handoff cache invalidations and official full resets. The old,
ambiguous `upstream_resets_sent` field is removed. Raw actions and Agent
decisions remain in `workspace/ACTION.md`, `logs/agent.log`, and
`agent_runs/<run_id>/events.jsonl`. Official JSON/video remain under `json/`
and `videos/`. If a run ends before the evaluator does, the runner first
requests a graceful stop so video writers can close; this does not turn an
interrupted episode into official success.

## Offline verification

The full integration test suite needs neither Isaac Sim nor an LLM:

When the champion `.venv` is installed, the suite also runs ten CPU-only
checks against the real wrapper and request handler, with fake inference and
no model weights. Without that environment the cross-environment check is
reported as skipped, not as a successful champion validation.

```bash
cd /data/yxy/EMERGE-Humanoid
/home/embodied/miniconda3/envs/phyagentos/bin/python \
  -m unittest discover -s scripts/Behavior_eval/checks -p 'test_*.py' -v
```

Verify one finished active episode from its artifacts:

```bash
/home/embodied/miniconda3/envs/phyagentos/bin/python -B \
  -m scripts.Behavior_eval.checks.verify_artifacts \
  --output-dir /data/yxy/outputs/Emerge/turning_on_radio \
  --require-success \
  --require-agent-result
```

## Current limits

- The evaluator is the unmodified pinned BEHAVIOR-1K checkout. EMERGE-specific
  observation publishing happens in the gateway, not in evaluator flags.
- R1Pro is currently the supported robot. Task selection is generic, while a
  task needs a champion checkpoint mapping. The runner does not load
  task-specific prompt files.
- The shadow gateway preserves the websocket payload but its synchronous side
  copy adds observation decoding and periodic disk I/O. Compare shadow results
  with the direct champion baseline before enabling active control.
