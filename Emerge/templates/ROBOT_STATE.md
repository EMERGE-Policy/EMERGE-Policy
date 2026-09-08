# Robot State

Auto-updated by Controller and/or side-loaded perception services.
This file stores the robot and scene runtime state in a structured format.

Agent usage:
- The robot state initially included in the agent context is only a snapshot.
- After every `execute_robot_action` call, the agent MUST use `read_file` to re-read `ROBOT_STATE.md` before checking task progress or choosing the next action.
- Do not rely on the earlier copy already present in the conversation context, because Controller may have updated the file after the action.

Notes:
- `robots.<robot_id>.connection_state` stores each robot's runtime connection health and reconnect metadata.
- `robots.<robot_id>.robot_pose` stores each robot's current pose state.
- `robots.<robot_id>.base_pose` stores the robot base pose for fixed-base manipulation reach checks when available.
- `robots.<robot_id>.nav_state` stores each robot's navigation/task runtime state.
- `objects` is the object-level world state used by current controller drivers.
- `grasp_constraints` stores active simulator gripper-to-target physical constraints when available.
- `runtime_targets` may expose executable target names and capability flags without exposing simulator ground-truth poses.
- `scene_graph.nodes[]` may include `frame`, `track_id`, and `last_seen_at` for semantic navigation.
- `scene_graph.edges[]` may include per-edge `confidence`.
- `map` may include `frame`, `resolution`, `origin`, `image_path`, and `zones`.
- `tf` stores summarized transform availability, not a full TF tree dump.

```json
{
  "schema_version": "Emerge.robot_state.v1",
  "scene_graph": {
    "nodes": [
      {
        "id": "red_apple",
        "class": "apple",
        "object_key": "red_apple",
        "center": {"x": 0.05, "y": 0.05, "z": 0.75},
        "size": {"x": 0.06, "y": 0.06, "z": 0.06},
        "confidence": 0.96,
        "frame": "map",
        "track_id": "track_apple_001",
        "last_seen_at": "2026-03-17T10:20:29Z"
      },
      {
        "id": "blue_cup",
        "class": "cup",
        "object_key": "blue_cup",
        "center": {"x": -0.10, "y": 0.03, "z": 0.78},
        "size": {"x": 0.08, "y": 0.08, "z": 0.12},
        "confidence": 0.94,
        "frame": "map",
        "track_id": "track_cup_003",
        "last_seen_at": "2026-03-17T10:20:28Z"
      }
    ],
    "edges": [
      {"source": "obj_red_apple", "relation": "ON", "target": "furniture_table", "confidence": 0.97},
      {"source": "obj_blue_cup", "relation": "ON", "target": "furniture_table", "confidence": 0.95},
      {"source": "obj_red_apple", "relation": "CLOSE_TO", "target": "obj_blue_cup", "confidence": 0.81}
    ]
  },
  "robots": {
    "go2_edu_001": {
      "connection_state": {
        "status": "connected",
        "transport": "ssh",
        "host": "192.168.1.23",
        "port": 22,
        "last_heartbeat": "2026-03-17T10:20:30Z",
        "last_error": null,
        "reconnect_attempts": 0
      },
      "robot_pose": {
        "frame": "map",
        "x": 1.23,
        "y": -0.45,
        "z": 0.0,
        "yaw": 1.57,
        "stamp": "2026-03-17T10:20:30Z"
      },
      "nav_state": {
        "mode": "navigating",
        "status": "running",
        "goal_id": "nav_goal_001",
        "target_ref": {"kind": "node", "id": "furniture_fridge", "label": "fridge"},
        "goal": {"x": 2.0, "y": 1.0, "yaw": 0.0},
        "path_progress": 0.62,
        "recovery_count": 1,
        "last_error": null,
        "relocalization_confidence": 0.91
      }
    },
    "desktop_pet_001": {
      "robot_pose": {
        "frame": "desk",
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "yaw": 0.0,
        "stamp": "2026-03-17T10:20:29Z"
      },
      "nav_state": {
        "mode": "idle",
        "status": "idle"
      }
    }
  },
  "grasp_constraints": {},
  "runtime_targets": {
    "cube_red": {"kind": "rigid_body", "body_type": "cube", "graspable": true, "environment_export": false},
    "cube_blue": {"kind": "rigid_body", "body_type": "cube", "graspable": true, "environment_export": false}
  },
  "map": {
    "frame": "map",
    "resolution": 0.05,
    "origin": {"x": 0.0, "y": 0.0, "z": 0.0},
    "image_path": "maps/home_demo.pgm",
    "stamp": "2026-03-17T10:20:25Z",
    "zones": [
      {"name": "kitchen", "center": {"x": 2.8, "y": 1.2, "z": 0.0}, "size": {"x": 1.5, "y": 1.0, "z": 2.4}}
    ]
  },
  "tf": {
    "map_to_odom": {"available": true, "stamp": "2026-03-17T10:20:30Z"},
    "odom_to_base_link": {"available": true, "stamp": "2026-03-17T10:20:30Z"}
  },
  "objects": {
    "red_apple": {
      "type": "fruit",
      "color": "red",
      "location": "table",
      "position": {"x": 5, "y": 5, "z": 0}
    },
    "blue_cup": {
      "type": "container",
      "color": "blue",
      "location": "table",
      "position": {"x": -10, "y": 3, "z": 0},
      "state": "empty"
    }
  }
}
```
