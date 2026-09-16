---
name: attach
description: Use geometry-driven motion primitives for coarse approach, clear-space positioning, and transport before or between fine model-policy phases.
metadata: {"Emerge":{"emoji":"🔧","always":true}}
---

# attach — Rule-Based Robot Control

## Attach Scope

Use attach for predictable geometric motion when the path and destination are sufficiently clear. Its main roles are coarse approach, open-space positioning, straight-line adjustment, and transport after a grasp has been established.

After `object_location` returns `found: true` for the current motion target,
attach is the default immediate handoff: coarse-approach a localized movable
object, or transport a held object toward a localized destination. Do not skip
straight to policy control unless the active embodiment profile explicitly
marks the task or object as policy-only.

Attach should normally stop in a useful pre-manipulation region and leave final visual alignment, grasping, placement, insertion, and other fine contact to the active model-policy backend. Do not use attach to guess a final wrist pose for an interaction whose success depends on camera view, contact direction, or continuous visual correction.

Every absolute target used by `move_to_pose` or `move_linear` must be derived
from current measured world-frame geometry. Valid sources are a successful
`object_location` result with `found: true` for a movable object, or an explicit
calibrated pose reported by the environment. Qualitative image descriptions,
`found: false`, failed localization, and model-estimated coordinates are not
motion targets.

## Choosing a Motion Primitive

### `move_to_pose`

Use for coarse empty-gripper motion through clear space when a direct pose target is appropriate.

```python
execute_robot_action(
    action_type="move_to_pose",
    parameters={
        "position_m": [x, y, z],
        "orientation_quat": [qx, qy, qz, qw],
        "steps": 40,
    },
    reasoning="Coarse positioning in clear space before the fine phase.",
)
```

### `move_linear`

Use when the path itself matters, especially for vertical lift or descent and for carrying an object through clutter. When orientation is omitted, the current wrist orientation is preserved.

```python
parameters={
    "position_m": [x, y, z],  # or "delta_m": [dx, dy, dz]
    "steps": 40,
    "settle_steps": 30,
}
```

### `set_gripper`

Use only when there is a clear reason to control the gripper independently.

```python
parameters={"command": "open", "steps": 30}  # or "close" / "opening_m"
```

A completed model-policy phase is not by itself a reason to open the gripper. Release according to the destination type:

- For an open supporting surface, release when the object is visibly over the intended footprint and already supported or low enough to settle without a disruptive fall.
- For a basket, compartment, drawer, or other containment target, release only when the object is visibly inside the intended region, below or safely past the relevant rim, and no longer depends on the gripper to remain there.

Being near, above, or within the broad footprint of a containment target is not sufficient. If placement remains ambiguous, continue the active fine policy phase or reassess from a safer view instead of opening the gripper. Do not use an independent close as a substitute for visually guided grasp alignment when finger placement is uncertain.

After a model-policy phase, re-read the robot state. If the intended open or close is only
partial, use the matching `set_gripper` command for 5 steps. For close, first
confirm the object is between the fingers and allow its width to keep `qpos`
above zero; otherwise return to the active policy backend. Do not add a separate lift test.

## Pose and Orientation

The commanded position is the end-effector target, not the object center. For coarse approach, aim for a nearby pre-manipulation pose rather than the final contact pose.

Preserve the current wrist orientation by default. `move_to_pose` requires an orientation, so reuse the current end-effector quaternion when available instead of inventing Euler angles. Change orientation only when the desired orientation is geometrically clear and does not belong to a view-sensitive fine manipulation phase.

If approaching with attach would require guessing how the wrist or camera should face the target, leave that approach to the active model-policy backend.

Thin, flat, toppled, or narrowly handled objects often require side access or a view-dependent grasp. Use attach only for a clearly useful pre-grasp region; let the active model policy establish the approach when choosing a reliable wrist posture would otherwise be guesswork.

## Confirming a Grasp Before Transport

Before transport, visually confirm the object is between the fingers; gripper
position or contact alone is insufficient. A normal clearance lift is
transport, not a grasp test.

If the object does not follow, open the gripper when appropriate, re-anchor on the intended object, and return the fine grasp to the active model-policy backend. Do not horizontally transport an object that may only be pinched, hooked, or resting against the gripper.

## Step Budget and Failed Motion Recovery

- Explicitly set `steps` for every motion action.
- Use about **20-30 steps** for coarse, collision-free motion and about **40-50 steps** for more careful geometric positioning.
- Do not request more than **100 steps** for one rule action.
- For `move_linear`, keep `steps + settle_steps <= 100` because both the path and final settling consume simulation steps.
- If a target pose does not converge, do not repeat the same pose with a larger budget. Use a shorter waypoint, reduce the displacement, preserve the current orientation, or change to an appropriate `move_linear` segment.
- A failed motion does not establish that its requested target was reached. Inspect the resulting state, but do not use a failed descent as evidence that an object is seated, supported, or ready for release.

After an unsuccessful or no-progress model-policy phase, do not generate speculative
`move_to_pose` or `move_linear` coordinates to re-anchor the robot. Rule-based
recovery is allowed only when a current reliable coordinate source supports the
new target and the path is clearly coarse and collision-free. In particular,
do not use attach to recover toward a cabinet, drawer, door, handle, or other
fixed articulated structure; leave that visual re-anchoring and interaction to
the active model-policy backend.

## Approach and Transport Geometry

For a pre-grasp approach, a small horizontal backoff of roughly **0.02–0.05 m** from the object, combined with a vertical offset of roughly **0.05 m above it**, can provide room for the active model policy to perform the final alignment. Treat these offsets as starting suggestions, not fixed requirements for every object shape or camera view.

If the target object is in a cluttered area, approach it by first reaching a
clear height, translating above the target, and then descending vertically to a
pre-grasp region; leave final contact to the active model policy.

Do not rely on a fixed absolute transit height. Choose clearance from the current geometry so the end-effector and any carried object pass comfortably above nearby obstacles and container rims.

When carrying an object through clutter, prefer three separate segments:

1. **Lift vertically** until the carried object has comfortable clearance.
2. **Translate horizontally** toward the destination while maintaining that clearance.
3. **Lower vertically** near the destination for the next fine manipulation phase.

Open, empty-gripper motion does not always require these three segments. The same lift-before-translate pattern is useful for retreating from a just-placed object without sweeping through it.

Treat objects already placed acceptably as obstacles to protect. Prefer lifting away before translating, and choose a path that does not sweep the gripper or carried object through an established task relation.

## Contact Recovery

Do not use attach as a pusher to nudge an ungrasped object into its final task relation. This includes pushing with an open gripper, a closed-but-empty gripper, or an unverified grasp. Such contact is especially risky near container rims, stove edges, and objects already placed correctly.

If an object is dropped, tipped, or released outside its target, re-anchor for a new visually guided grasp or placement attempt. After two attach motions fail to improve the same recovery state, stop constructing nearby contact waypoints and change the setup or action family.

## Key Principles

- Use attach to support the active model-policy backend, not to replace fine visual manipulation.
- Preserve wrist orientation unless there is a clear geometric reason to change it.
- Stop at a useful pre-manipulation region rather than forcing a final contact pose.
- Confirm the intended object is visibly secured and settle a partial close before transport without requiring zero finger opening.
- Set a moderate explicit step budget and change the motion plan after non-convergence.
- Prefer vertical-horizontal-vertical transport when carrying through clutter.
- Release on an open surface only when landing is controlled; release into a containment target only after the object is visibly inside and past the relevant rim.
- Do not recover final placement by pushing an ungrasped object with attach.
