---
name: vla
description: Use the pi0.5 Vision-Language-Action policy for visually guided, contact-rich manipulation when the target is visible and adaptive alignment, grasping, placing, insertion, or articulated interaction is needed.
metadata: {"Emerge":{"always":true}}
---

# vla — Vision-Guided Manipulation

Use `vla_execute` for fine manipulation that benefits from continuous visual correction. It may perform a local phase after attach has approached the target, or handle the approach itself when wrist pose, camera view, and contact direction must adapt together.

## Before Execution

Identify the current manipulation target and intended local change. Treat the
complete task only as context; do not use it as the current VLA instruction.
Preserve already acceptable task relations when continuing a multi-object task.

Use VLA when the target is visible and the next phase involves adaptive alignment or contact. If the end-effector is far from an ordinary object and no pose-sensitive interaction is required, use attach for the coarse approach first.

### Phase-Local Instructions

Every `vla_execute` instruction must describe exactly the next unfinished
visual-contact phase, even when the current PLAN subgoal combines grasp,
transport, and placement. Never copy the Mission, mention a later object, or
combine grasp and final placement in one instruction.

Keep it to one short imperative sentence: include only the current action,
active object, and destination when needed. Omit reasoning, observations,
coordinates, progress history, and later steps.

- Grasp phase: `grasp the second moka pot`.
- Placement phase: `place the held second moka pot on the stove`.

A single continuous articulated phase may include its visual approach, contact,
and actuation, such as `open the top drawer`, but it must not absorb a later
object-manipulation phase.

For a later object in a multi-object task, omit every already completed movable
object from the VLA instruction. During grasping, name only the active movable
object and the local grasp intent. During placement, name only the held active
object, its current destination, and the release intent. Do not restate the
complete task to remind VLA of prior progress: each VLA call starts from the
current observation and may otherwise restart with the first named object.

## Step Budget

`step` is an execution budget, not an estimate that the requested semantic action will be finished within that many steps.

- Use about **30-40 steps** as an initial chunk for a simple local grasp or correction after a useful attach approach.
- Use about **60-70 steps** when VLA must also establish an ordinary visual approach or perform pose-sensitive placement or insertion.
- Use **220-300 steps in one continuous call** for a single fixed articulated-mechanism phase such as opening or closing a drawer, cabinet door, or microwave door. The approach, handle contact, and actuation belong to one continuous visually guided behavior; do not interrupt it at 110 steps merely to verify progress.

`step_completed` means only that the requested budget was exhausted and control returned. It does not mean that the grasp, placement, insertion, or full task is complete. `goal_reached` is the explicit task-success result.

Execution stops when any of the following occurs:

- The step budget is exhausted.
- The environment ends.
- The task goal is reached earlier.

After every VLA chunk, reassess the visible state and choose one of these transitions:

1. **Continue VLA** only while the same local contact phase is still in progress and continuous visual correction remains useful. Do not continue VLA into a later clear-space phase merely because the preceding chunk made progress.
2. **Hand off to attach** when the intended local phase is visibly established. A verified secure grasp ends the VLA grasp phase; if measured destination geometry and a clear transit path are available, lift and transport with rule actions. A placement or insertion is not established merely because the object is near the destination or still held above it.
3. **Stop and re-anchor** when VLA selects the wrong object, disturbs an already acceptable relation, loses the target, creates unsafe contact, or makes no meaningful progress. Reposition or change the approach before trying again.

Do not release an object merely because a VLA call returned `step_completed`. Do not treat a larger budget as recovery from wrong-target behavior or repeated no-progress behavior. After **two consecutive no-progress chunks** for the same target and intent, change the setup or action strategy rather than issuing the same call again.

For fixed articulated structures, recovery remains vision-guided. Do not ask
the object-location specialist to measure the cabinet, drawer, door, or handle,
and do not invent a `move_to_pose` or `move_linear` target after an unsuccessful
VLA call. Continue from a clearly useful visible state with VLA, or stop when no
safe useful action remains.

## Multi-Object and Long-Horizon Instructions

A full task may naturally require several VLA chunks, but each chunk remains
bound to the current PLAN subgoal. Returning at a budget boundary can occur in
the middle of one grasp or one final placement; it is not permission for one
VLA instruction to absorb later transport or next-object subgoals.

When the current object is completed, explicitly shift attention to the next target before continuing. Prefer an approach that avoids revisiting or colliding with objects already placed acceptably. If the VLA begins manipulating a completed object or a destination container instead of the intended movable object, stop and re-anchor rather than allowing a longer rollout to reinforce the mistake.

## Example

```python
execute_robot_action(
    action_type="vla_execute",
    parameters={
        "instruction": "pick up the red block",
        "step": 10
    },
    reasoning=(
        "The end effector is near the visible red block, and the grasp requires "
        "fine visual alignment."
    ),
)
```

If this call returns `step_completed` while the correct red block remains securely held and is still being purposefully aligned, continue VLA. If the grasp is clearly established and the next motion is open-space transport, hand off to attach. If another object is grasped instead, stop and re-anchor on the red block.
