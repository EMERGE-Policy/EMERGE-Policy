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

## Target identity

For an ambiguous target, combine:

`object class + visual signature + original task relation`

Use shape, color, and visible details together; lighting can change a single
color cue. Bind the target using its original task relation and track the same
instance after motion. Re-establish identity when continuity is lost rather
than substituting another same-class object.

### Object appearance hints

- **Radio receiver:** the asset observed in the current radio experiments has
  a red body, pale trim, and a dark circular speaker area. Combine these cues
  with the task's table and room relation; do not identify the receiver from
  a red patch alone. Assets and orientations may differ between instances, so
  confirm the current appearance rather than requiring this color scheme.

### Observable state hints

- **OmniGibson toggle marker:** when the object's actual toggle-state marker
  is identifiable and visible, **red means off** and **green means on**. This
  is the simulator's marker convention, not a general rule for appliance LEDs.
- Distinguish the marker from the radio's red casing, speaker details,
  reflections, and other colored pixels. A bright red spot is not evidence
  that the radio is powered on, and an arbitrary green patch is not an on-state
  marker either.
- If the marker is cropped, occluded, too small, or cannot be confidently
  identified, report the toggle state as uncertain. Missing visual evidence
  does not prove that the object is off. Visual state evidence does not
  replace the official success signal defined below.

### Identity and state information for subagents

The `object_location` and `task_verification` specialists do not automatically
inherit this profile. Make each delegated request self-contained: include the
target's relevant appearance, original task relation, and exclusions. When
requesting toggle-state verification, explicitly include the red-off/green-on
marker convention and require uncertain when the marker is not identifiable.
Ask for evidence about the requested object, not a generic claim that a light
is illuminated.

When a visual `done_criterion` in PLAN.md depends on this marker, include the
relevant marker convention there too; the background visual monitor receives
that criterion, not this profile. These hints inform perception and verification
requests; they do not change the champion's exact `effective_prompt` contract.

## Default task workflow

Use the native EMERGE division between measured coarse motion and visually
guided fine interaction. Base positioning and end-effector positioning are
different: a nearby target in the head image does not establish that a hand
is in a useful pre-manipulation pose. Track task-level milestones in PLAN.md;
choose navigation and arm-motion details at execution time from current evidence.

For a target that needs coarse approach, the default handoffs are:

`assess base position -> navigate if needed -> localize target -> attach/IK pre-approach if needed -> confirm actual pose -> VLA fine interaction`

1. **Establish a useful base position.** Use current observations and, when
   needed, `task_verification` to assess target visibility, the operating side,
   and visible obstructions. Use the `lightnav` skill when base travel or
   repositioning is needed; skip it when the base is already suitably placed.
   A navigation stop or a visual claim of being nearby is not proof of arm
   reach, body clearance, or end-effector readiness.
2. **Measure the current motion target.** When the target is visible and a
   coarse arm approach is needed, delegate to `object_location`; reading its
   skill alone does not run localization. Include the target identity and
   task relations, relevant support/obstacle geometry, and require results in
   `robot_base`. Use only a successful, unambiguous `found: true` result with
   measured geometry. Localization needs its configured SAM3 service. A
   service failure or `found: false` is not a usable motion target. After base
   motion, do not reuse old base-relative coordinates without re-establishing
   them from fresh observations; re-localize a target that has moved.
3. **Bring the end effector to a pre-manipulation region.** After successful
   localization, use the existing `attach` skill and IK-backed `move_to_pose`
   with `coarse: true` for a needed, clear-space coarse approach. Choose the
   arm from the current measured geometry and task requirements; neither hand
   has default priority. Keep the intended subsequent interaction accessible.
   This is the default next control action when reliable geometry and a clear
   path are available and the hand is not already suitably positioned. Choose
   a measured pre-contact point outside the object's occupied volume; do not
   target its centre, guess a control's location, or push it with a rule action.
   Treat current wrist orientation as a preference, not a precise grasp pose.
   Accept a useful partial approach; do not repeat IK merely to reduce the
   remaining distance. After checking the actual state, leave fine alignment
   and contact to VLA. Use `move_linear` only for a separately justified short
   straight motion, not as another approximate attempt at the same point.
   Follow the R1Pro parameters below rather than copying another robot's
   action schema. Joint limits and IK convergence are not collision checks.
4. **Confirm the actual hand pose.** Re-read synchronized ROBOT_STATE.md and
   compare the measured end-effector pose with the intended pre-approach.
   Use current camera evidence when contact, occlusion, or clearance is
   ambiguous. A solved IK target or completed command does not establish
   physical arrival. Keep base suitability and hand readiness separate when
   deciding whether a manipulation milestone is reached.
5. **Hand off fine interaction to VLA.** Let the champion handle final visual
   hand alignment, grasping, placement, or actuation. The local focus belongs in
   PLAN.md and action reasoning; every VLA call still uses the exact runtime
   `effective_prompt`, not a rewritten phase prompt. The policy retains full
   body control and may reposition after handoff; no base lock is introduced.
6. **Reassess progress and recover.** After an action, inspect the resulting
   state and relevant visible change, not only whether execution succeeded.
   `step_completed`, `state_synced`, and an evaluator that is still running
   are not evidence of task progress. Follow the planning skill's retry and
   recovery rules; repeating VLA with a larger budget is not itself recovery.

Skip redundant IK when current evidence already places the hand in a useful
pre-manipulation region. If reliable geometry or a clear coarse path cannot
be established, or choosing the approach requires fine visual alignment,
improve the observation or let VLA handle that approach when justified.
Fixed articulated contact remains visually guided; do not use attach to pull
or actuate a mechanism. State the evidence/reason for bypassing a default
handoff in the plan or action reasoning; VLA's whole-body capability alone is
not a reason to skip a feasible measured coarse approach. Do not invent
coordinates to force the route or repeat a failed IK target with more steps.

## Skill responsibilities

- `lightnav`: base travel and repositioning through `navigate_vln`.
- `object_location`: measured target and clearance-relevant geometry, not motion.
- `attach`: measured arm pre-approach and clear-space transport through the
  supported IK motion primitives; stop before fine contact.
- `task_verification`: visible scene state and task relations, not metric
  coordinates, guaranteed reachability, or the official success signal.
- `vla`: final alignment and contact-rich interaction with the champion policy.

## Supported actions

Examples illustrate parameter syntax, not a preferred hand.

### `vla_execute`

Calls the champion PI_BEHAVIOR policy, which can control whole-body motion and
visually sensitive contact. The gateway forwards all 23 action dimensions
unchanged, including base x/y/yaw velocity. There is no navigation/manipulation
control-mode switch. VLA can perform its own approach and repositioning, but
apply the default workflow above when a measured coarse arm approach is
appropriate. LightNav first is optional, not a required phase.

This evaluation entrypoint uses the official episode instruction. Copy
`robots.behavior_r1pro.effective_prompt` exactly into every `vla_execute` call;
keep the current local phase in `PLAN.md` and the action reasoning rather than
rewriting the policy prompt. `step` is a positive policy action budget no
larger than `robots.behavior_r1pro.max_action_steps`. Use bounded chunks and
inspect progress between them.

### `navigate_vln`

Use the built-in `lightnav` skill for natural-language base navigation:

```json
{"instruction": "Approach the requested object", "max_steps": 60}
```

The driver uses fresh head-camera frames and holds the other joints at their
latest measured positions. It returns on this call's budget, a model stop, or
an error. Healthy repeated calls within an episode retain LightNav history.
The Agent chooses navigation instructions, budgets, and recovery actions.
`max_steps` is this call's budget, bounded by
`max_action_steps`; the official evaluator still owns the episode step limit.
LightNav requires its external service. It does not provide automatic obstacle
avoidance or prove arrival from `visible=true` or a model stop.

### `adjust_posture`

Optionally adjust body posture or camera views with a relative joint motion:

```json
{"joint_deltas_rad": {"torso_joint4": 0.1}, "max_steps": 40}
```

Use measured `body_posture` in ROBOT_STATE.md: valid names are `torso_joint1`
through `torso_joint4`, `left_arm_joint1` through `left_arm_joint7`, and
`right_arm_joint1` through `right_arm_joint7`. Each delta is limited to ±0.25
rad, with at most 0.01 rad per step and 10 settling steps, within URDF limits.
Base velocity is zero; other joints and grippers hold position. The head
camera is fixed to `torso_link4`, and wrist cameras to their grippers. Link
origins are not collision volumes. No environment/self-collision check or
automatic posture adjustment is provided. Inspect the synchronized actual
pose and views afterward; target tracking error above 0.05 rad is a failure.

### `move_to_pose`

Move one end effector toward a reachable measured position:

```json
{"side": "right", "position_m": [0.55, -0.18, 0.92], "coarse": true, "steps": 40}
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
  "standoff_m": 0.12,
  "coarse": true,
  "steps": 40
}
```

For a bounding box, the driver finds the surface toward the selected arm and
requests a point 8-15 cm before it, not the occupied object centre. `auto`
selects the nearer arm for a bounding-box target; there is no preferred hand.
For an explicit `position_m`,
`side` must be `left` or `right`. `orientation_quat` uses XYZW order. When it
is omitted, precise motion preserves the current wrist orientation.

`coarse` is an optional boolean for `move_to_pose` only (default false), not a
global control mode. With true, one bounded solve seeks a reachable position
toward the requested point; centimetre-level position or orientation error is
not by itself a failure. The supplied orientation, or the current orientation
when omitted, is a soft preference. No global nearest-point guarantee is made.
The result includes a `Coarse approach plan` with `requested_position_m`,
`planned_reachable_position_m`, and `remaining_distance_m` in `robot_base`.
The endpoint is URDF forward kinematics, not measured arrival, target identity,
or proof of a usable grasp. Re-read the actual state and views after motion.
Do not retry just to eliminate the reported remaining distance. A failed solve,
no useful progress, invalid coordinates, or an endpoint inside the supplied
object box still prevents dispatch; no collision-free path is guaranteed.
With false or omitted, the original exact-pose IK checks remain unchanged.

Both `move_to_pose` forms and `move_linear` accept optional `steps` and
`settle_steps`, including the generic attach examples. `steps` is a positive
integer specifying the minimum motion duration, not an early-stop budget;
omitting it uses the duration planned from joint/path limits. A request shorter
than the safe trajectory is extended, never sped up or truncated before the
target. `settle_steps` is a non-negative integer (default 0) that adds actual
final-position hold commands after motion. Motion plus settling must fit the
driver's `primitive_max_steps` limit (default 240), otherwise the action is
rejected before dispatch. These timing fields do not select coarse IK or relax
coordinate checks or joint limits, and do not establish collision clearance.

### `move_linear`

Move one end effector along a straight Cartesian path to a reachable measured
position or relative displacement:

```json
{"side": "right", "delta_m": [0.0, 0.0, 0.08], "steps": 30, "settle_steps": 20}
```

Use this only for a short final approach, lift, or retreat through visibly
clear space. Provide exactly one of absolute `position_m` or relative
`delta_m`; `orientation_quat` is optional. Timing follows the `steps` and
`settle_steps` contract above; settling holds the final command rather than
adding more Cartesian displacement.

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

`gateway_status` is read-only. `attach` is a generic EMERGE skill that composes
the supported `move_to_pose`, `move_linear`, and `set_gripper` actions; it is
not a separate driver action. `follow_arc` is not implemented by this driver.

Every local primitive is bounded by official joint limits and a per-step joint
change limit. Before dispatch, the driver compares URDF forward kinematics with
the observed end-effector position. It refuses the action if the coordinate
conventions disagree. Switching back to the OpenPI policy invalidates only its
cached actions and rolling-inpainting seed so it replans from the post-motion
observation. This also applies after LightNav. The task ID, current subtask
stage, stage-voting history, policy state and counters are preserved; a handoff
does not restart the task at stage 0. Consecutive VLA calls without intervening
local motion keep their normal action cache. Only an official new-episode
reset or a task change performs the original full stage reset. The champion
service must advertise `invalidate_action_cache`; an old service is rejected
at handoff rather than falling back to a full reset. Restart the updated
champion service before running this integration. External motion and fresh
action planning still differ from uninterrupted champion baseline execution.

## Result and success semantics

An action result ending in `step_completed` says only that its action budget
was consumed and synchronized with the simulator. It does not mean the task
succeeded. The sole authoritative success signal is
`evaluation.success == true` together with `evaluation.phase == "completed"`.
Stop immediately when `evaluation.done` is true. Treat `evaluation.phase ==
"error"` as an evaluator failure. While the episode is running,
`evaluation.steps` is unavailable; `runtime_progress.gateway_action_steps`
is a live diagnostic counter, not the official score. Check `state_synced`
before judging the result of the most recent action.
