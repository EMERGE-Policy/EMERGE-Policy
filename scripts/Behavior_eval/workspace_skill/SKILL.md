---
name: vla
description: Drive the BEHAVIOR-1K turning_on_radio π0.5 checkpoint through the native behavior_r1pro driver, using its exact frozen instruction and bounded action budgets.
metadata: {"Emerge":{"always":true}}
---

# vla — BEHAVIOR-1K Turning On Radio

This workspace shadows the generic manipulation `vla` skill. In this episode
`vla_execute` does **not** take a phase-local instruction: the official π0.5
server fixes its prompt when it starts, so the only instruction that reaches the
policy is the exact task sentence. A rephrased instruction is rejected before it
is dispatched.

## Scope

Use this skill only while `robots.behavior_r1pro.mode` is
`controlled_pi05_gateway` in `ROBOT_STATE.md`. The official BEHAVIOR evaluator
owns the simulator, the episode reset, the termination decision and the score.

## The only control action

```json
{
  "action_type": "vla_execute",
  "parameters": {
    "instruction": "Turn on the radio receiver that's on the table in the living room.",
    "step": 300
  },
  "reasoning": "Continue the active BEHAVIOR episode, then inspect official progress."
}
```

Rules:

1. `instruction` must be that exact English sentence, character for character.
   Do not translate it, shorten it, split it into phases, or append
   observations. Every rewrite is rejected.
2. `step` is the number of policy action replies this call may consume. It is an
   action budget, not a probability of success and not a count of model
   forward passes. It must be a positive integer no larger than
   `robots.behavior_r1pro.max_action_steps`.
3. Use one bounded chunk per call and re-inspect the official state between
   chunks. A long budget is not a substitute for verifying progress.

`gateway_status` is available as a diagnostic action, but it is not part of the
control loop.

Do not use `move_to_pose`, `move_linear`, `set_gripper`, `follow_arc`,
`attach`, `navigate_vln` or object localization in this episode: the driver
exposes no such action, and the evaluator only accepts the policy's own actions.

## Bounded loop

1. Read `ROBOT_STATE.md` and confirm `evaluator_connected` and
   `upstream_connected` are true, and that `evaluation.done` is not yet true.
2. Dispatch one `vla_execute` chunk.
3. Read `ROBOT_STATE.md` again. Confirm `state_synced` is true; if it is false,
   the policy's last actions may not have been applied to the simulator yet, so
   the reported state can be one step stale.
4. Inspect the newest camera manifest at
   `artifacts/observations/observation.json`. The views are uncalibrated, so use
   them only for qualitative judgement, never for metric geometry.
5. Repeat while `evaluation.done` is false and progress is still being made.

## Success and stopping

- The **only** authoritative success signal is
  `robots.behavior_r1pro.evaluation.success == true` together with
  `evaluation.phase == "completed"`.
- `step_completed` means the requested budget was consumed. `episode_finished`
  means the evaluator disconnected. Neither means the radio is on.
- `evaluation.phase == "error"` is an evaluator failure, not a task failure.
- Stop dispatching as soon as `evaluation.done` is true, then report the
  official result.
- If the evaluator has terminated without success, report the task as not
  completed. Do not retry the same chunk, and do not claim success.
- Do not start another gateway, controller or simulator from the shell.
