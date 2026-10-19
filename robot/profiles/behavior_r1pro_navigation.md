# BEHAVIOR R1Pro — Agent-guided navigation and body observation

Driver: `behavior_r1pro`

The official evaluator owns simulator time and the R1Pro robot. Physical
actions use the normal EMERGE Controller and gateway. The simulator is paused
between actions. The champion policy supplies the gateway protocol metadata,
but must not control the robot during this navigation mission.

## Observation and body state

Read `artifacts/observations/observation.json` for the current synchronized
`head`, `left_wrist`, and `right_wrist` RGB-D views and their calibrated
transforms in `robot_base` (+X forward, +Y left, +Z up). View images through
the existing `task_verification` specialist; use `object_location` when
measured object geometry is needed and its SAM3 service is available.

`robots.behavior_r1pro.body_posture` in ROBOT_STATE.md contains measured joint
angles, URDF joint limits, and body-link transforms. Link origins are not body
dimensions, obstacle clearance, or collision detection. Head-camera visibility
alone does not establish that the torso or arms can pass an obstacle.

The head camera is fixed to `torso_link4`; it has no independent pan/tilt
joint. The wrist cameras are fixed to the grippers. Moving the torso changes
the head view and both arms; moving an arm changes that wrist view. Torso
joints 1–2 rotate about local +Y, joint 3 about local -Y, and joint 4 about
local +Z (also recorded as `joint_axes_local` in body state). Refer
to measured transforms rather than assuming a joint motion is a pure camera
rotation. The current posture is not automatically changed when navigating.

## Actions

Read the built-in `lightnav` skill. Choose action order,
instruction, call budget, and any subagent use autonomously.

### `navigate_vln`

```json
{"instruction": "Approach the requested object", "max_steps": 60}
```

The driver uses fresh head frames, executes bounded base motion, and returns
on this call's budget, a model stop, or an error. Other joints hold their
latest measured positions. Multiple calls within the same episode keep the
LightNav history unless a connection or episode change requires a reset;
the result reports `history_continued`. The example budget is not a schedule
or required chunk size. Select it from current evidence.

### `adjust_posture`

```json
{"joint_deltas_rad": {"torso_joint4": 0.1}, "max_steps": 40}
```

This is relative joint motion, not a named safe pose or an IK target. Valid
joint names are `torso_joint1` through `torso_joint4`, `left_arm_joint1`
through `left_arm_joint7`, and `right_arm_joint1` through `right_arm_joint7`.
Use the latest measured angles and URDF limits in ROBOT_STATE.md. Each
requested delta must be within ±0.25 rad; the driver interpolates commands
at at most 0.01 rad per step and holds the target for 10 settling steps.
It rejects a plan that exceeds its budget or URDF joint limits. The base
command is zero throughout; grippers and unrequested joints hold position.
This does not guarantee an immobile base under external forces.

Choose a small change only when current evidence supports room for that
motion. No self-collision, environment-collision, or force detection is
provided by this action. Both camera direction and body clearance may change.
After the action use the newly synchronized multi-view observation and
reported **actual** body posture; issued commands do not prove that the
requested pose was reached. A requested joint that remains more than 0.05 rad
from its target is reported as a failed posture action. The driver does not
choose a posture for you.

### `gateway_status`

Read-only connection status. The driver rejects VLA, object manipulation,
and gripper actions in this navigation-only experiment. Complete-task runs
use `behavior_r1pro.md` and allow the same navigation skill alongside VLA.

## Budget and completion

`remaining_action_steps` is the shared episode budget remaining for navigation
and posture actions; each call must fit it. It is distinct from a call's
`max_steps`. Saved traces append all calls, and runtime progress exposes
`navigation_calls` and `posture_calls` with observed outcomes.

A model `stop` or `visible=true` is a prediction, not measured arrival. You
may inspect, correct, and navigate again. Finish with the evidence for the
requested position and any uncertainty; do not claim navigation success from
these flags alone. This experiment does not measure turning on or manipulating
the target. It does not provide automatic obstacle avoidance.
