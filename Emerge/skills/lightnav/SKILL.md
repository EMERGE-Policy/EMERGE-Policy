---
name: lightnav
description: Follow a natural-language navigation goal with LightNav when the active robot supports navigate_vln.
---

# LightNav navigation

Read EMBODIED.md and the current ROBOT_STATE.md. Use this skill only when the
active embodiment provides `navigate_vln`. Choose whether and when navigation
is useful for the current task; there is no required navigation-first sequence.

Call `execute_robot_action` with `action_type="navigate_vln"` and a non-empty
navigation `instruction` plus a positive `max_steps`. Choose the instruction
and budget from the task, current evidence, and embodiment limits. This
navigation instruction is separate from any fixed task instruction required
by the VLA checkpoint.

The driver handles fresh camera frames and bounded base actions inside each
call. At action boundaries decide whether to continue navigation, change the
instruction, inspect the scene, use VLA, or finish. Multiple navigation calls
are allowed; there is no prescribed number of calls or observation schedule.
The driver reports whether LightNav history continued between calls.

Use the existing `task_verification` subagent to examine actual images when
visual evidence matters, or `object_location` when measured geometry is
needed and its services are available. Reading an image manifest or a trace's
`visible=true` is not viewing an image. An unavailable service is not evidence
of an empty scene. Captured observations may be stale after physical actions.

The embodiment may also provide posture actions; their availability, joint
limits, and parameters belong in EMBODIED.md and ROBOT_STATE.md. No posture
change is automatic. Joint limits and small increments do not establish
collision clearance. This skill does not provide automatic obstacle avoidance.

`step_budget` means the current navigation call consumed its budget.
`model_stop_candidate` returns control to you; neither it nor `visible=true`
proves arrival or completion of the overall task. Decide the next action from
the task and current observations. Follow the embodiment's authoritative
success signal and any restrictions imposed by a navigation-only experiment.

When handing control to VLA, follow its skill and the active embodiment's
instruction contract. The navigation skill does not select a VLA stage,
rewrite a fixed policy prompt, or lock any part of the VLA action.
