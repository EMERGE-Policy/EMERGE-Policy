# BEHAVIOR-1K R1Pro π0.5 Gateway

## Identity

- **Robot:** R1Pro humanoid in OmniGibson
- **Task:** `turning_on_radio`
- **VLA:** official BEHAVIOR π0.5 checkpoint
- **Driver:** `behavior_r1pro`
- **Mode:** `controlled_pi05_gateway`

## Supported Action

The official BEHAVIOR evaluator owns the simulator, episode reset, task
termination, metrics, and video recording. This driver owns only the gateway
between the evaluator and the official OpenPI π0.5 server, so the only action it
accepts is `vla_execute`.

```json
{
  "action_type": "vla_execute",
  "parameters": {
    "instruction": "Turn on the radio receiver that's on the table in the living room.",
    "step": 300
  }
}
```

Rules the driver enforces:

- `instruction` must be that exact sentence. The π0.5 server fixes its prompt at
  startup, so a rephrased instruction is rejected locally instead of silently
  running the original one. `prompt` is accepted as a legacy alias.
- `step` must be a positive integer no larger than
  `robots.behavior_r1pro.max_action_steps`. It is a count of policy action
  replies, not a number of model forward passes: the OpenPI server caches an
  action horizon internally.
- `parameters` must be an object, and `step` must be present.

`gateway_status` is also accepted as a read-only diagnostic. It never consumes
the policy budget and is not part of the control loop.

Motion primitives (`move_to_pose`, `move_linear`, `set_gripper`, `follow_arc`)
and `attach` are not available for this embodiment; the evaluator only applies
the policy's own actions.

## Result semantics

| Result | Meaning |
|---|---|
| `VLA execution finished: step_completed` | The requested budget was consumed **and** the simulator was confirmed to apply the actions. |
| `VLA execution finished: episode_finished` | The evaluator disconnected while the action was in flight. |
| `Interrupted: ...` | The Controller cancelled the action. The queue records this as `cancelled`. |
| `Failed: ...` | Validation, readiness, timeout, upstream failure, or an unconfirmed simulator state. The queue records this as `failed`. |

`step_completed` never means the task succeeded.

## State published to the Agent

`robots.behavior_r1pro` in `ROBOT_STATE.md` additionally exposes:

- `state_synced` — whether the last action was confirmed applied. While it is
  `false`, the reported evaluation may be one step stale.
- `applied_steps` — the evaluator's step counter after synchronization.
- `max_action_steps`, `execution_timeout_s`, `progress_timeout_s`,
  `state_sync_timeout_s` — the active budgets.
- `gateway_status` — the raw gateway lifecycle status, separate from the
  driver's own `status`.
- `evaluation` — the live contents of the official evaluator status file.
- `last_disconnect` — why the evaluator connection ended, when it did.

The authoritative result remains the JSON written by the BEHAVIOR evaluator;
`evaluation.success` together with `evaluation.phase == "completed"` is the only
success signal. `evaluation.phase == "error"` is an evaluator failure, not a
task failure.

## Synchronization and timeouts

The evaluator applies an action and only then sends the next observation, so a
new observation frame is direct evidence that the actions reached the simulator.
The driver waits for that frame, and falls back to the evaluator's step counter
or a terminal status when the episode ends with the last action.

- `execution_timeout_s` bounds a single `vla_execute` call.
- `progress_timeout_s` bounds the wait between two confirmed policy replies.
- `state_sync_timeout_s` bounds the wait for simulator confirmation.

Exceeding any of them fails the action instead of returning a success-shaped
result.

## Cameras

Camera snapshots are published to `artifacts/observations/observation.json`
together with the images they reference. Every published revision carries
`observation_seq`, `episode_id` and `captured_at`, and each image path is
revision-specific so a reader can never mix two frames. The views have no
simulator camera extrinsics and their intrinsics are approximations: they must
not be used as calibrated 3D measurements.
