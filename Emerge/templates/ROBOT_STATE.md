# Robot State

Auto-updated by Controller and/or side-loaded perception services.
This file stores the robot runtime state in a structured format.

Agent usage:
- The robot state initially included in the agent context is only a snapshot.
- After every `execute_robot_action` call, the agent MUST use `read_file` to re-read `ROBOT_STATE.md` before checking task progress or choosing the next action.
- Do not rely on the earlier copy already present in the conversation context, because Controller may have updated the file after the action.

Notes:
- `robots.<robot_id>.connection_state` stores each robot's runtime connection health and reconnect metadata.
- `robots.<robot_id>.robot_pose` stores each robot's current pose state.
- `robots.<robot_id>.base_pose` stores the robot base pose for fixed-base manipulation reach checks when available.
- `robots.<robot_id>.nav_state` stores each robot's navigation/task runtime state.
- `grasp_constraints` stores active simulator gripper-to-target physical constraints when available.
- `runtime_targets` may expose executable target names and capability flags without exposing simulator ground-truth poses.
- `map` may include `frame`, `resolution`, `origin`, `image_path`, and `zones`.
- `tf` stores summarized transform availability, not a full TF tree dump.

```json
{
  "schema_version": "Emerge.robot_state.v1",
  "robots": {}
}
```
