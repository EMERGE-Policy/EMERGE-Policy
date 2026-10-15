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

The default ports are EMERGE gateway `8000`, VGGT `8001`, SAM3 `8002`, and
champion OpenPI `8011`. BEHAVIOR uses calibrated RGB-D, so the normal path does
not need VGGT.

The general runner contains no task name or task instruction. Those values
come from `dev/behavior_agent_eval.json`, or from the `--task-name` and
`--task-instruction` overrides. Optional task guidance belongs in
`tasks/<task_name>.md`; `tasks/turning_on_radio.md` is the first example.

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
runner requires the exact task instruction for every champion VLA call,
supervises the native Controller and controlled gateway, and reads success
only from the official result JSON. A completed action chunk is not itself
task success.

## Offline verification

The full integration test suite needs neither Isaac Sim nor an LLM:

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
  task needs a champion checkpoint mapping and may optionally need its own
  `tasks/<task_name>.md` guidance.
- The shadow gateway preserves the websocket payload but its synchronous side
  copy adds observation decoding and periodic disk I/O. Compare shadow results
  with the direct champion baseline before enabling active control.
