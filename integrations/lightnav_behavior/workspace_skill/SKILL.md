---
name: lightnav-behavior-preview
description: Optionally inspect LightNav-0's navigation suggestions from BEHAVIOR head-camera frames without changing the champion policy's actions.
---

# LightNav navigation preview (read-only)

This skill is available only in the BEHAVIOR LightNav shadow experiment. The
champion policy alone controls the R1Pro. You decide whether and when a
navigation preview is useful; the evaluator runner does not schedule it.

When the current scene presents a meaningful navigation subgoal, call
`preview_navigation` with a short instruction limited to that subgoal, for
example "Approach the table with the red radio and stop within manipulation
reach." Do not ask LightNav to press a button, turn on the radio, or perform a
whole manipulation task.

`max_frames` is a small number of fresh head-camera observations to inspect,
not a robot-action budget. The tool keeps the LightNav session across calls in
the same episode and resets it when the episode changes. The returned waypoint
is `[forward_m, left_m, yaw_rad]` in the robot's current local frame. It is a
prediction, not a velocity command or a measured target position.

The tool cannot move, stop, or reset the robot. `stop=true` is only LightNav's
suggestion, not proof of arrival or official task success. After the call,
compare its trace with fresh camera evidence; report uncertainty when the
target is occluded, already within reach, or the preview makes no progress.
It is valid not to call this skill if navigation is not useful in the episode.
