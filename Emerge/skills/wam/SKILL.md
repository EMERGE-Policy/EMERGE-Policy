---
name: wam
description: Use the Cosmos Policy WAM backend for visually guided LIBERO manipulation with local best-of-N action-chunk search.
---

# wam — Cosmos Policy Manipulation

Use wam_execute for visually guided manipulation when the LIBERO environment
provides third-person and wrist observations and the WAM service is healthy.
WAM may take over after a verified coarse approach, or handle the approach
itself when alignment, contact, grasping, placement, insertion, or articulation
must adapt visually.

## Task and Phase Conditioning

Identify exactly one current manipulation phase, active target, and intended
local change. The evaluator preserves the exact benchmark text separately as
task_instruction. Your phase_instruction describes only the current phase.

The locked conditioning_mode controls what Cosmos encodes:

- task: the original benchmark instruction.
- phase: the current phase_instruction.
- task_with_phase: both texts in a stable combined format.

The complete task text is model context, not permission to execute every
remaining phase in one call. Keep the active phase in PLAN.md and reassess when
the budget returns or execution is interrupted.

Use one short imperative phase_instruction naming the active object and local
goal. Never mention a later object or combine grasp and final placement.

- Grasp: grasp and lift the second moka pot securely.
- Placement: place the held second moka pot on the stove and release it.
- Wrong: grasp the second moka pot and place it on the stove.

A continuous articulated phase may include its visual approach, contact, and
actuation, such as opening one drawer, but must not absorb a later object phase.

## Geometry-Assisted Handoffs

Rule motion is optional support, not a compulsory state machine. Before a WAM
grasp, use at most one non-contact pre-position only when current geometry,
end-effector distance, and a collision-free offset endpoint are all known.
Preserve wrist orientation unless a measured orientation is required, and stop
outside the contact envelope. WAM owns final alignment, descent, closure, and
lift.

After WAM returns, check ROBOT_STATE.md first. Consider rule transport only
after the intended object is visibly verified as securely held and lifted, and
only when destination geometry and a clear-space path are current. Lift to
clearance, translate through free space, and stop above or beside the
destination. WAM owns final descent, placement alignment, and release.

Keep WAM in control when a geometry gate is missing, contact direction is
uncertain, the object is close to an obstacle, or WAM is making purposeful
contact progress. Never invent a target-directed rule pose after an incomplete
WAM attempt.

## Step Budget

step is a raw action7 budget for one call, not proof that a semantic action will
finish within it.

- Use about 40–60 steps for a local grasp or correction after a useful approach.
- Use about 80–110 steps for visual approach, placement, or insertion.
- Use 220–300 continuous steps for one fixed articulated phase such as opening
  or closing a drawer, cabinet door, or microwave door.
- Outside that fixed-mechanism case, keep one call at or below 110 steps.

step_completed means only that the budget was exhausted. goal_reached means the
official task-success condition was observed. interrupted means the visual
monitor or controller ended the call early. environment_done and error are not
success.

After every WAM call, immediately re-read ROBOT_STATE.md:

1. Continue WAM only while the same contact phase is active, the correct target
   is being affected, and purposeful progress is visible.
2. Hand off only when the local phase is visibly established and a measured,
   collision-free rule target exists.
3. Re-anchor when WAM selects the wrong object, loses the target, disturbs an
   acceptable relation, creates unsafe contact, or makes no progress.

Do not release an object because a budget ended. After two consecutive
no-progress calls for the same target and intent, change setup or strategy.

## Candidate Search

At each replan the server returns configured candidate chunks and scores. The
Emerge-side planner deterministically selects the highest score; the agent does
not choose candidates manually. Scores compare short-horizon actions and do not
prove semantic progress or task completion.

## Multi-Object Tasks

Handle one selected object at a time. Omit already completed movable objects
from phase_instruction. Once one object's contact-sensitive grasp-to-placement
phase begins, do not insert speculative rule moves or independent gripper
commands. Shift the active target explicitly before acting on the next object.

Example call:

    execute_robot_action(
        action_type="wam_execute",
        parameters={
            "phase_instruction": "grasp and lift the red block",
            "step": 48,
        },
        reasoning="The active phase requires visual grasp alignment.",
    )
