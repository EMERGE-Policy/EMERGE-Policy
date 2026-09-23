# Unitree G1 LightNav-0 + SONIC Embodiment

## Identity

- **Name:** G1 SONIC navigation robot
- **Type:** Unitree G1 humanoid in MuJoCo
- **Driver:** `lightnav_sonic_g1`
- **Navigation policy:** LightNav-0 VLN
- **Locomotion:** NVIDIA SONIC Kinematic Planner + WBC

## Sensors

- **Navigation camera:** G1 original `head_camera`, published as `ego_view`.
- **Heading:** world yaw derived from SONIC `g1_debug.base_quat`.
- The current `g1_debug` stream does not provide a trustworthy measured world X/Y pose. Never invent X/Y coordinates or metric path progress.

## Supported Actions

### `navigate_vln`

Parameters:

```json
{
  "instruction": "move next to the table and stop",
  "max_steps": 120,
  "control_hz": 4.0
}
```

- `instruction` is required and describes only the current navigation subgoal.
- `max_steps` is a control-tick budget, not a distance. At 4 Hz, 120 steps are about 30 seconds.
- The first-stage driver caps one action at 120 steps; reassess the observation before issuing another action.
- `control_hz` is optional; use 4 Hz unless a test explicitly requires another rate.
- The driver uses only LightNav's first local waypoint and applies conservative SONIC velocity limits.

## Navigation Protocol

1. Before acting on a multi-step request, create a PLAN with a distinct navigation subgoal and an observable done criterion.
2. Use `navigate_vln` for locomotion. Do not use manipulation actions in this embodiment.
3. After every action result, re-read `ROBOT_STATE.md` and inspect the latest observation before updating the plan.
4. `Navigation action completed` means the low-level action ended. It does not prove that the semantic destination was reached.
5. LightNav `stop=true` ends the action but still requires task-level visual verification.
6. A step-budget result requires reassessment. Do not report success merely because `max_steps` was exhausted.
7. Do not call the LIBERO-oriented `object_location` specialist for metric localization: this single camera is not yet published with calibrated multi-view geometry.

## Physical and Safety Constraints

- Only one process may bind the SONIC planner publisher at `tcp://*:5556`.
- Never run the standalone LightNav bridge runner or keyboard teleop while this Controller is active.
- Missing/stale camera or yaw, invalid LightNav output, cancellation, and communication errors must stop locomotion.
- A normal navigation stop is SONIC locomotion `IDLE`; it must not terminate the SONIC deployment process.
- First-stage validation is simulation-only.

## Recovery Guidance

- If the target is not visible or the robot makes no useful progress, keep the main navigation subgoal active and create a localized recovery branch.
- Do not repeat the same instruction indefinitely. After two no-progress attempts, change the viewpoint or report that an explicit scan/turn primitive is required.
- The original downward-looking `head_camera` is a known limitation. Treat ambiguous views as uncertainty, not success.
