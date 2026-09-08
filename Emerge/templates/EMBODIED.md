# EMBODIED.md Template

This file describes one embodied robot instance.
The agent reads the runtime `EMBODIED.md` to understand which actions the robot supports and how to parameterize them.

Use this template to understand the structure and meaning of each section.
Do not keep concrete robot-specific values here in production; put those in `robot/profiles/*.md` and let the controller copy the matching profile into the robot workspace.

## Identity

Describe what this robot is.
Typical fields:
- **Name**: human-readable robot name
- **Type**: robot category, e.g. quadruped, arm, desktop pet, mobile manipulator
- **Driver/Profile**: optional implementation identifier

## Degrees of Freedom or Sensors

Use one or both of these sections depending on the robot type.

### Degrees of Freedom

List movable joints, ranges, and what they control.

| Joint | Range | Description |
|-------|-------|-------------|
| example_joint | example range | what this joint does |

### Sensors

List the robot's sensing capabilities when perception matters.

- **Camera**: optional
- **LiDAR**: optional
- **Odometry**: optional
- **Microphone**: optional

## Supported Actions

This is the most important section for the agent.
List the actions the robot can safely accept, their parameters, and what they do.

| Action | Parameters | Description |
|--------|-----------|-------------|
| `example_action` | `arg1, arg2` | Example action description |

## Physical Constraints

Describe hard limits and safety boundaries.
Examples:
- **Workspace bounds**
- **Max payload**
- **Max reach**
- **Collision policy**
- **Speed limits**

## Connection

Describe how the robot is reached by the controller.
This section is static capability/configuration, not runtime status.
Examples:
- **Transport**
- **Host**
- **Port**
- **User**
- **Auth**
- **Reconnect Policy**
- **Health Check**

Runtime connection state belongs in `ROBOT_STATE.md` under `robots.<robot_id>.connection_state`.

## Runtime Protocol

Optionally document how this robot maps into shared runtime state.
Examples:
- **Connection channel**: `robots.<robot_id>.connection_state`
- **Pose channel**: `robots.<robot_id>.robot_pose`
- **Navigation channel**: `robots.<robot_id>.nav_state`
- **Health owner**: usually `controller.py`

## Navigation Protocol

Use this section when the robot participates in navigation.
Typical items:
- **Robot-state schema**: usually `Emerge.robot_state.v1`
- **Scene graph assumptions**
- **Safety distance**
- **Relocalization support**
- **ROS2 bridge support**

## Authoring Rules

- Keep this file specific to one robot profile.
- Put concrete robot values in `robot/profiles/*.md`, not in the template.
- Keep runtime status out of this file; runtime status belongs in `ROBOT_STATE.md`.
