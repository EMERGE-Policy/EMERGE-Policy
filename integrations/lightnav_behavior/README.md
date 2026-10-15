# LightNav-0 for BEHAVIOR R1Pro

This integration is currently at the **read-only inference** stage. The official
LightNav source is pinned at `third_party/lightnav-0`; model weights belong
outside Git, for example in `/data/yxy/checkpoints/lightnav-0`. The existing
`lightnav_sonic_emerge` integration remains the G1/SONIC prototype. Neither
integration currently sends LightNav waypoints to BEHAVIOR's R1Pro.

The `probe` reads the `head` view from a BEHAVIOR observation manifest, encodes
it as JPEG, and uses the already tested LightNav WebSocket protocol client. It
does not write an experiment result or advance the simulator.

Validate a captured observation without a model server:

```bash
cd /data/yxy/EMERGE-Humanoid
python -m integrations.lightnav_behavior.probe \
  --manifest /path/to/workspace/artifacts/observations/observation.json \
  --instruction "Go to the radio on the table and stop" \
  --dry-run
```

After installing the pinned source in a separate Python 3.11 environment and
preparing the compatible checkpoint, start the official server. The base
installation provides the `hf` backend; `vllm_local` needs the separate
`[vllm]` extra. The BEHAVIOR head image is square, so `--aspect_mode keep`
preserves its aspect ratio during this first probe:

```bash
/data/yxy/envs/lightnav/bin/lightnav-serve \
  --task vln \
  --backend hf \
  --aspect_mode keep \
  --model_path /data/yxy/checkpoints/lightnav-0 \
  --port 8051
```

Rerun the probe without `--dry-run` after the service is ready. This is a
single-frame protocol and waypoint check, not a navigation success test.

The next integration boundary is an R1Pro-specific adapter from LightNav's
robot-local `[forward_m, lateral_left_m, delta_yaw_rad]` waypoint to the
official 23-D action. The evaluator runs at 30 Hz, and its base controller
scales normalized input `[-1, 1]` to x/y velocities within `[-0.75, 0.75] m/s`
and yaw within `[-1, 1] rad/s`. A closed-loop controller must execute a
bounded number of simulator steps, obtain a fresh head frame, and replan. Do
not directly reuse the SONIC movement-command adapter for this robot.
