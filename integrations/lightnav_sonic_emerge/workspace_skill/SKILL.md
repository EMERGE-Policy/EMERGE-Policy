---
name: lightnav-sonic-navigation
description: Navigate a Unitree G1 in simulation using LightNav-0 visual-language waypoints and SONIC locomotion.
always: true
---

# LightNav-0 + SONIC Navigation

Use `execute_robot_action` with `action_type="navigate_vln"` for a navigation subgoal.

## Invocation

Provide a non-empty instruction limited to the current navigation goal:

```json
{
  "action_type": "navigate_vln",
  "parameters": {
    "instruction": "move next to the table and stop",
    "max_steps": 120,
    "control_hz": 4.0
  },
  "reasoning": "The current plan focus is reaching the table before any later task."
}
```

`max_steps` is a control-tick budget. At 4 Hz, 120 ticks are approximately 30 seconds; it is not 120 metres and does not guarantee completion.
The configured first-stage ceiling is 120 ticks per action; use a new verified action instead of one unbounded call.

## Execution Loop

1. For a multi-step task, write a PLAN whose current subgoal has an observable navigation done criterion.
2. Invoke only the current navigation phase. Do not include future manipulation in the LightNav instruction.
3. After the action, immediately re-read `ROBOT_STATE.md` and inspect current visual evidence.
4. Treat LightNav stop or action completion as a request to verify, not as proof of arrival.
5. Advance PLAN only when the done criterion is visibly established.
6. When the budget is exhausted or progress is absent, retain the current subgoal and create a localized recovery branch.
7. Do not issue the same no-progress navigation call more than twice. Change viewpoint or report the need for an explicit scan/turn recovery primitive.

Do not use the LIBERO object-location path for this first-stage single uncalibrated G1 camera. Do not start shell-based bridge runners: the active EMERGE Controller owns the SONIC planner port and action lifecycle.
