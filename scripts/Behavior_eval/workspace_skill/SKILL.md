---
name: vla
description: Execute the BEHAVIOR-1K turning_on_radio task with the R1Pro using bounded π0.5 navigation, calibrated object localization, safe arm primitives, and official evaluator feedback.
metadata: {"Emerge":{"always":true}}
---

# BEHAVIOR-1K Turning On Radio

This episode uses the native `behavior_r1pro` driver. The simulator is frozen
while you reason and advances only when `execute_robot_action` is running. Make
one physical decision at a time, then re-read `ROBOT_STATE.md` and the newest
observation manifest.

## Invariants

- Every `vla_execute` call must use exactly:
  `Turn on the radio receiver that's on the table in the living room.`
- `step_completed` is action completion, not task success.
- Only `evaluation.success == true` with `evaluation.phase == "completed"` is
  authoritative success. Stop when `evaluation.done` is true.
- Geometry may be used only when `object_location` returns `found: true` in the
  `robot_base` frame. Never infer metres directly from an image.
- Reinspect the scene after every physical action because wrist cameras and
  object visibility change with the robot.

## Phased route

Maintain the current phase in `PLAN.md` and follow this order.

### 1. Coarse navigation

Confirm `evaluator_connected` and `upstream_connected`. Inspect all current
views. Call `vla_execute` with 200-400 steps. Continue with another chunk only
if the new images show meaningful progress toward the living-room table and
radio. Two chunks without visible approach progress trigger recovery rather
than an identical third call.

### 2. Calibrated localization

When the radio is clearly visible and plausibly within reach, call:

```json
{
  "agent_name": "object_location",
  "task": "Using every calibrated current view, locate the radio receiver on the living-room table and, only if it is independently visible in at least two views, its power button or power control. Return measured geometry in the observation coordinate frame; do not guess an unobserved control."
}
```

Require `status: success`, `found: true`, and `frame: robot_base` before using
any returned coordinate. A single head RGB-D view may provide the radio's coarse
visible-surface box when its identity is visually unambiguous. The power control
still requires independent confirmation in at least two views. A radio
bounding-box center is not automatically a button coordinate.

### 3. Measured pre-contact approach

If the power control has a measured bounding box, pass the returned
`bbox_world_min_m` and `bbox_world_max_m` directly to `move_to_pose`, with
`side: auto` and `standoff_m: 0.12`. The driver deterministically selects the
nearer arm, approaches the near box surface rather than its occupied centre,
and preserves the live wrist orientation. Do not copy the box centre into
`position_m`. A short `move_linear` may follow only when the target is measured
and the complete segment is visibly clear.

If the driver rejects IK or reports an FK/state mismatch, do not alter the
coordinate or retry the same primitive. Use a short VLA chunk to reposition the
whole body, refresh observations, and localize again.

### 4. Fine contact

Use `vla_execute` with 80-160 steps for final visual alignment and pressing the
control. The checkpoint still receives the exact full task sentence; never
replace it with a phase-local instruction. Inspect official state and fresh
camera views immediately afterward.

### 5. Verify or recover

When visual evidence is ambiguous, delegate `task_verification` with the
self-contained question: whether the radio now visibly appears powered on and
what visible evidence supports that conclusion. Its result guides recovery but
does not override the official evaluator.

Recovery choices:

- Localization service error or unconfirmed segmentation: continue with
  bounded VLA control; do not repeatedly call the failed service.
- Radio still far away: use another coarse VLA chunk.
- Radio visible but control occluded: use a short VLA repositioning chunk, then
  obtain fresh localization.
- Primitive rejected as unreachable: reposition with VLA and relocalize.
- Failed final contact with a good approach pose: use one fresh short VLA
  contact attempt after inspection, not an identical long rollout.

Do not start servers, controllers, or simulators from shell tools inside the
episode. The runner owns those processes.
