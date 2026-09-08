---
name: planning
description: Decompose a task into a static main line of subgoals and execute it reactively, using a branch stack to absorb failures.
always: true
---

# Planning

For any multi-step embodied task, track progress in `PLAN.md`. It holds the whole state; you drive the transitions. Update it only with `update_plan`; `write_file`, `edit_file`, and `exec` must not modify `PLAN.md`.

## Planning Model

- **Main Line**: an ordered list of coarse subgoals (milestones). Do not pre-expand them into concrete robot actions; choose the primitive and action count when executing the current phase.
- **Active Target**: the specific object or mechanism the current subgoal intends to affect. Keep it explicit so motion of a different object is not mistaken for progress.
- **Intended Change**: the task-level change expected from the current subgoal, such as a bowl being lifted or a drawer becoming open enough to use.
- **Done Criterion**: a practical, observable condition showing that the intended change has been established.
- **Pointer**: the one main-line subgoal currently in progress.
- **Branch Stack** (LIFO): temporary recovery subgoals pushed when the main-line approach fails or the scene regresses. An empty stack means execution is on the Main Line.

## Execution Loop

1. **Plan** — on a new task, call `update_plan` with `update_kind="new_mission"`, the Mission, and a short Main Line of coarse milestones.
2. **Recover context** — read `PLAN.md` first every turn and identify the current focus: the Branch Stack top when non-empty, otherwise the Pointer subgoal.
3. **Bind the target** — keep the current Active Target and Intended Change in mind before choosing an action.
4. **Act** — work only on the current focus. Choose rule actions or `vla_execute` at execution time according to the active embodiment guidance and the needs of the current phase.
   When the plan has later subgoals, a `vla_execute` instruction must name only
   the current focus and must not include those future subgoals.
5. **Refresh robot state** — immediately after every `execute_robot_action` result, use `read_file` to read the current `ROBOT_STATE.md`. The copy in the original context is stale once the controller has executed an action. Perform this refresh before any PLAN update, visual verification, subagent call, or next robot action.
6. **Check terminal state** — apply any environment-specific terminal-success rule first. A confirmed full-mission success ends the execution loop immediately; do not continue reasoning about subgoals or call another tool.
7. **Reassess** — only when the full mission is not complete, compare the observed change with the current Done Criterion.
8. **Update** — advance or pop when the subgoal is done; otherwise update Retries and recovery state before acting again.

## Practical Completion Judgment

An action reported as completed only means that its execution ended; it does not automatically complete the current subgoal.

Judge completion at the task level rather than demanding perfect geometry. Mark a subgoal done when the intended target and relation are clearly established and there is no obvious contradictory outcome. Small offsets, imperfect centering, or a non-ideal pose should not trigger extra correction when the result is already functionally adequate. Avoid disturbing an acceptable result merely to make it look perfect.

Keep the subgoal active when the intended effect is clearly absent. Motion of the wrong object is regression, not progress. When the result is genuinely ambiguous, prefer a brief reassessment or a low-risk verification action before committing to a more disruptive next phase.

Examples of practical completion include a bowl remaining on a plate without being perfectly centered, a bottle resting securely on a cabinet without being perfectly upright, or a drawer being open far enough for the next manipulation without being fully extended.

## Retry and Recovery

- Increase `Retries` when an action fails, completes without meaningful progress toward the Intended Change, or affects the wrong target.
- `Retries` never decreases for a retained semantic subgoal. Do not rename a subgoal or use replan merely to reset its counter.
- Once a subgoal reaches **`Retries >= 2`**, do not repeat the same approach. Change the action strategy, push a useful recovery subgoal, or rewrite the Main Line.
- If the Branch Stack grows deeper than 2, simplify the recovery plan or report that intervention is needed rather than adding more nested recovery steps.
- While the Branch Stack is non-empty, work only on its top. When its Done Criterion is met, pop it and leave the main-line Pointer where it was. Reassess the interrupted main-line subgoal before continuing.

## Plan Update Rules

Every `update_plan` call supplies the complete desired state:

- Use `progress` for normal status, Pointer, Retries, push, and pop changes. It cannot rewrite the Main Line.
- Use `replan` only when intentionally rewriting the Main Line.
- Use `new_mission` only for a different task, or to restart the same task after its previous plan completed.
- Keep `branch_stack` in bottom-to-top order; the final item is the active top.
- A completed plan uses `Pointer = len(Main Line) + 1`.

## Hybrid Task Example

For "open the drawer and put the bowl inside", a coarse Main Line may be:

1. **Open the drawer** — Active Target: drawer handle; Intended Change: the drawer becomes open enough to use. A pose-sensitive approach and opening phase may be handled by VLA.
2. **Grasp the bowl** — Active Target: bowl; Intended Change: the bowl is carried with the gripper. Attach may perform the coarse approach and VLA the fine grasp.
3. **Move the bowl near the drawer** — Active Target: open drawer; Intended Change: the grasped bowl reaches a useful pre-placement region. Attach may perform the clear-space transport.
4. **Place the bowl inside** — Active Target: bowl and drawer interior; Intended Change: the bowl remains inside the drawer. VLA may perform the fine placement; a functionally adequate placement is sufficient.

The action examples are selected only when each phase becomes active; they are not a fixed action script embedded in the Main Line.
