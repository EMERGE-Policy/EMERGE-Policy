# AC-WM: Cosmos3 candidate evaluation

AC-WM adds action-conditioned Cosmos3 prediction to EMERGE's planning tools.
It runs alongside the OpenPI VLA and Cosmos Policy WAM paths. WAM generates and
scores its own policy candidates inside `wam_execute`; AC-WM evaluates explicit
skill alternatives supplied by the main agent through `ac_wm_select`.

## Execution flow

1. The main agent supplies a current recovery subgoal and 2–8 skill candidates.
2. The controller captures one observation. Each skill produces a control
   proposal from that paused simulator state, without executing it.
3. Cosmos3-Edge predicts an action-conditioned video for each distinct proposal.
4. A separately configured VLM receives the current observation, plan context,
   candidate controls and ordered forecast frames in one comparison request.
5. The selector chooses a positive top-scoring candidate. A positive-score tie
   is resolved using the main agent's explicit `planner_priority` values.
6. The controller executes the selected prefix or calls the selected skill,
   according to `dispatchMode`. The main agent then reads the updated state.

The tool guidance uses AC-WM for recovery after an unsuccessful grasp or
placement, at most once per subgoal. Ordinary `vla_execute`, `wam_execute` and
geometric actions remain direct calls. This usage policy is prompt guidance;
the explicit `ac_wm_select` call is the runtime entry point.

## Candidate and execution semantics

Supported candidate skills are `vla_execute`, `move_to_pose`, `move_linear`,
`set_gripper` and `follow_arc`. In a WAM-only configuration, geometric
candidates remain available and VLA candidates respect the existing backend
restriction. `wam_execute` retains its own candidate search; it is not an
AC-WM proposal source.

| Setting | Behavior |
| --- | --- |
| `dispatchMode: evaluated_prefix` (default) | Replay the selected action rows up to the evaluated execution horizon. |
| `dispatchMode: skill` | Run the selected skill with the planner's original parameters. A closed-loop skill can produce different controls from its preview. |
| `onNoPositive: reject` (default) | Return without executing when no candidate has a positive usable forecast score. |
| `onNoPositive: planner_priority` | Run the planner's first-choice skill when forecast selection has no winner; record this as fallback, separately from a selected forecast. |

Equivalent action prefixes are deduplicated. If all alternatives collapse to
one prefix, the request asks the planner for distinct controls. The adapter
keeps each candidate's real proposal length, rounds the Cosmos prediction
horizon down to a multiple of four, and never pads actions. The judge sees the
execution-prefix endpoint separately from later hypothetical frames.

LIBERO controls use seven dimensions. The Cosmos adapter preserves normalized
translation controls, converts axis-angle rotation to rotation-6D, maps the
gripper to `1=open`, and applies the pinned LIBERO action statistics. The
conditioning image concatenates the third-person and wrist views after a
180-degree rotation of each view.

## Implementation

| Component | Source |
| --- | --- |
| Candidate contracts and selection | [`protocol.py`](../../Emerge/ac_wm/protocol.py), [`selector.py`](../../Emerge/ac_wm/selector.py) |
| Skill proposals and dispatch | [`actions.py`](../../Emerge/ac_wm/actions.py), [`subagent.py`](../../Emerge/ac_wm/subagent.py) |
| Cosmos3 input and transport | [`cosmos_adapter.py`](../../Emerge/ac_wm/cosmos_adapter.py) |
| Joint visual scoring | [`vlm_judge.py`](../../Emerge/ac_wm/vlm_judge.py) |
| Paused-state controller previews | [`robot/mujoco_simulation/ac_wm.py`](../../robot/mujoco_simulation/ac_wm.py) |
| Persistent Cosmos3 service | [`cosmos3_server.py`](../../external_model_server/cosmos3_server.py) |

See [configuration and operation](operation.md) for installation, service
startup, a selection example and the regression-test command.
