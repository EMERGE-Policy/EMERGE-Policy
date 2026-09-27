# BEHAVIOR R1Pro embodiment

Driver: `behavior_r1pro`

The official BEHAVIOR evaluator owns simulator time, reset, termination, score,
and video. The driver grants individual action budgets through the evaluator's
websocket connection. Simulator time is frozen while the Agent is reasoning.

## Coordinate and state contract

All measured positions used by the actions below must be in `robot_base`, the
frame named by `artifacts/observations/observation.json`. In this frame +X is
forward, +Y is left, and +Z is up. Never copy image pixels into a metric action.

The observation manifest contains three calibrated RGB-D views (`head`,
`left_wrist`, and `right_wrist`), exact intrinsics for the official full
resolution images, metre-valued `depth_linear` arrays, base-relative camera
transforms, and structured R1Pro proprioception.
`robots.behavior_r1pro` in `ROBOT_STATE.md` contains the live evaluator and
gateway status. Read both files again after every physical action.

## Supported actions

### `vla_execute`

Use the official π0.5 policy for coarse whole-body navigation and visually
sensitive contact:

```json
{
  "instruction": "Turn on the radio receiver that's on the table in the living room.",
  "step": 300
}
```

The instruction must be exactly that sentence. `step` is a positive policy
action budget no larger than `robots.behavior_r1pro.max_action_steps`. Use
bounded chunks and inspect progress between them.

### `move_to_pose`

Move one end effector toward a reachable measured position:

```json
{"side": "left", "position_m": [0.55, 0.18, 0.92]}
```

`side` is `left` or `right`. `position_m` must come from a successful
`object_location` result in `robot_base`, or be a clearance point derived from
that result and the current end-effector position. `orientation_quat` in XYZW
order is optional. When it is omitted, IK constrains position only.

### `move_linear`

Move one end effector along a straight Cartesian path to a reachable measured
position:

```json
{"side": "right", "position_m": [0.61, -0.16, 0.89]}
```

Use this only for a short final approach or retreat through visibly clear
space. It has the same `side`, `position_m`, and optional `orientation_quat`
parameters as `move_to_pose`.

### `set_gripper`

```json
{"side": "left", "command": "open", "steps": 15}
```

Provide exactly one of `command` (`open` or `close`) and `opening_m` (0.0 to
0.1 metres). `steps` is optional. Turning on the radio normally does not
require a grasp, so do not close a gripper without a concrete reason.

### `hold_position`

```json
{"step": 1}
```

This repeats current joint and gripper positions with zero base velocity. Use
it only for synchronization or diagnosis. Reasoning itself does not require a
hold because the simulator is paused between actions.

`gateway_status` is read-only. `follow_arc`, `attach`, and `navigate_vln` are
not implemented by this driver.

Every local primitive is bounded by official joint limits and a per-step joint
change limit. Before dispatch, the driver compares URDF forward kinematics with
the observed end-effector position. It refuses the action if the coordinate
conventions disagree. Switching back to π0.5 resets its action chunk so the
policy replans from the post-primitive observation.

## Result and success semantics

An action result ending in `step_completed` says only that its action budget
was consumed and synchronized with the simulator. It does not mean the task
succeeded. The sole authoritative success signal is
`evaluation.success == true` together with `evaluation.phase == "completed"`.
Stop immediately when `evaluation.done` is true. Treat `evaluation.phase ==
"error"` as an evaluator failure.

## Required task route

1. Confirm evaluator and OpenPI connectivity and inspect the newest views.
2. Use one 200-400 step `vla_execute` chunk for coarse navigation. Repeat only
   when the new head view shows meaningful approach progress.
3. Once the radio is clearly visible and plausibly within arm reach, call the
   `object_location` specialist. Ask it to locate the radio and, when visibly
   separable in at least two views, its power control. The returned frame must
   be `robot_base`.
4. If a control position is measured and reachable, choose the arm on the same
   side. Derive a pre-contact point 8-15 cm toward the current end effector from
   the measured control. Use `move_to_pose` for that clearance point. Use a
   short `move_linear` only when its entire path is visibly clear.
5. Use a short 80-160 step `vla_execute` chunk for the final visual alignment
   and button contact. Re-read official state and fresh images.
6. Call `task_verification` when visible evidence of the radio state is useful,
   but never replace the official evaluator success signal with its answer.

If localization is unavailable or cannot distinguish the control, keep using
bounded VLA chunks; do not invent coordinates. If IK rejects a target, use a
short VLA repositioning chunk, obtain a fresh observation, and localize again.
After two VLA chunks with no visible approach progress, do not blindly repeat a
third: localize, change the recovery action, or report the observed blocker.
