---
name: task-verification
description: Delegate visual state and outcome verification to the registered task_verification subagent, either before an action whose need depends on the current scene or after an embodied action. Use for visible states such as held, released, placed, open, closed, exposed, or obstructed.
metadata: {"Emerge":{"always":true}}
---

# Task Verification Delegation

Use `delegate_subagent` with `agent_name="task_verification"` to inspect current visible task state or check whether the latest embodied action achieved its visible goal. The specialist only observes and verifies; robot control and recovery remain with the main Agent.

## Initial Scene Check

Before the first robot action, use this specialist when an articulated action is
only conditionally needed—for example, a target is described as being inside a
drawer but the drawer may already be open and the target exposed. Ask whether
the structure already has the needed visible state and whether the target is
visible or visibly obstructed. Keep the plan conditional until this result: if
the target is already exposed, skip the articulated action and localize the
movable target. Do not add this check when current scene state cannot change the
next action or an embodiment profile forbids `task_verification`.

## Visual Monitor Completion

The visual monitor uses the same multi-camera verification capability while an action is running. If `execute_robot_action` reports that the visual monitor interrupted the action because the current subgoal is achieved, that report is already the required visual verification.

In that case, do **not** call `delegate_subagent` with `agent_name="task_verification"` for the same subgoal. Directly mark the subgoal done in `PLAN.md`, advance to the next focus, and continue execution. Calling the verification subagent again would duplicate a completed judgment.

## Form the Verification

State the exact visible conditions that define success. Phrase every requested
condition as something that must be true for success; do not mix in failure
alternatives whose false value would actually support success. Verify a
meaningful action phase when it may be complete, rather than treating
`step_completed` or a returned tool call as success.

Before transport, ask whether the object is visibly secured, unsupported, and
lifted with the gripper. Finger contact, apparent enclosure, or nonzero gripper
opening while the object still rests on its support is not a verified grasp.

```json
{
  "agent_name": "task_verification",
  "task": "Verify whether the apple is inside the basket and has been released by the gripper."
}
```

Request only conditions that can be judged from current images. Do not ask this specialist for coordinates, motion plans, or another action. Use `object_location` in a separate delegation when updated world geometry is needed.

## Use the Result

Read `output.outcome` and the evidence in `output.predicates`:

- `achieved`: every required visible condition is satisfied. Continue or finish the corresponding task phase.
- `not_achieved`: at least one required condition is visibly false. Use the evidence to choose a correction or recovery.
- `uncertain`: at least one condition cannot be established from the current views. Improve the observation or report uncertainty; do not claim success.

Use `output.scene_context` as qualitative context only. Re-run verification after a corrective action because each result describes one current scene snapshot.
