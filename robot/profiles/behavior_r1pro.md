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

Use the champion PI_BEHAVIOR policy for coarse whole-body navigation and
visually sensitive contact:

This evaluation entrypoint uses the official episode instruction. Copy
`robots.behavior_r1pro.effective_prompt` exactly into every `vla_execute` call;
keep the current local phase in `PLAN.md` and the action reasoning rather than
rewriting the policy prompt. `step` is a positive policy action budget no
larger than `robots.behavior_r1pro.max_action_steps`. Use bounded chunks and
inspect progress between them.

### `move_to_pose`

Move one end effector toward a reachable measured position:

```json
{"side": "left", "position_m": [0.55, 0.18, 0.92]}
```

`side` is `left` or `right`. `position_m` must come from a successful
`object_location` result in `robot_base`, or be a clearance point derived from
that result. When the result describes the target object or control with a
bounding box, prefer the deterministic pre-contact form:

```json
{
  "side": "auto",
  "bbox_world_min_m": [0.48, -0.12, 0.82],
  "bbox_world_max_m": [0.62, 0.02, 0.96],
  "standoff_m": 0.12
}
```

The driver selects the nearer arm when `side` is `auto`, finds the box surface
toward that arm, and stops 8-15 cm before it. This avoids commanding the centre
of an occupied object as an end-effector target. For an explicit `position_m`,
`side` must be `left` or `right`. `orientation_quat` uses XYZW order. When it
is omitted, both motion primitives preserve the current wrist orientation.

### `move_linear`

Move one end effector along a straight Cartesian path to a reachable measured
position or relative displacement:

```json
{"side": "right", "delta_m": [0.0, 0.0, 0.08]}
```

Use this only for a short final approach, lift, or retreat through visibly
clear space. Provide exactly one of absolute `position_m` or relative
`delta_m`; `orientation_quat` is optional. This matches the generic attach
skill contract.

### `set_gripper`

```json
{"side": "left", "command": "open", "steps": 15}
```

Provide exactly one of `command` (`open` or `close`) and `opening_m` (0.0 to
0.1 metres). `steps` is optional. Close a gripper only when the current plan
and visible geometry provide a concrete reason.

### `hold_position`

```json
{"step": 1}
```

This repeats current joint and gripper positions with zero base velocity. Use
it only for synchronization or diagnosis. Reasoning itself does not require a
hold because the simulator is paused between actions.

### `set_control_mode`

The episode starts in whole-body `navigation` mode:

```json
{"mode": "navigation"}
```

After the visual verifier confirms that the target is in manipulation range
and no further base translation is needed, switch at the action boundary:

```json
{"mode": "manipulation"}
```

In `manipulation` mode the gateway forces the first three entries of every
official 23-D policy action (base x/y/yaw velocity) to zero. Trunk, both arms,
and both grippers remain controlled by the OpenPI policy. The transition resets
OpenPI so
the next VLA call replans from the current observation. Check
`control_mode` and `base_motion_enabled` in `ROBOT_STATE.md` before
contact. Return to `navigation` only when fresh verification shows that the
target is outside manipulation range and base repositioning is required.

`gateway_status` is read-only. `attach` is a generic EMERGE skill that composes
the supported `move_to_pose`, `move_linear`, and `set_gripper` actions; it is
not a separate driver action. `follow_arc` and `navigate_vln` are not
implemented by this driver.

Every local primitive is bounded by official joint limits and a per-step joint
change limit. Before dispatch, the driver compares URDF forward kinematics with
the observed end-effector position. It refuses the action if the coordinate
conventions disagree. Switching back to the OpenPI policy resets its action chunk
so the policy replans from the post-primitive observation.

## Result and success semantics

An action result ending in `step_completed` says only that its action budget
was consumed and synchronized with the simulator. It does not mean the task
succeeded. The sole authoritative success signal is
`evaluation.success == true` together with `evaluation.phase == "completed"`.
Stop immediately when `evaluation.done` is true. Treat `evaluation.phase ==
"error"` as an evaluator failure.
