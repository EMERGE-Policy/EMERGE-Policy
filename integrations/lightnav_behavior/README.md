# LightNav-0 for BEHAVIOR R1Pro

This integration includes a **read-only Agent-selected shadow preview**, an
**Agent-controlled navigation and body-observation runner**, and a separate
**fixed active-navigation handoff ablation**. The official
LightNav source is pinned at `third_party/lightnav-0`; model weights belong
outside Git, for example in `/data/yxy/checkpoints/lightnav-0`. The existing
`lightnav_sonic_emerge` integration remains the G1/SONIC prototype. Both active
paths execute through the existing R1Pro driver and gateway; the shadow
preview never changes simulator actions.

## Normal EMERGE skill and complete-task execution

The active navigation skill now lives at `Emerge/skills/lightnav/SKILL.md`,
discovered by the standard EMERGE skill loader. Its former
`workspace_skill_active/SKILL.md` copy has been migrated; complete-task runs
do not install a navigation-only skill override. This integration directory
keeps the model client helpers and R1Pro adapter, not another Agent loop.
The passive preview's `workspace_skill/` remains a separate read-only tool.

`scripts/Behavior_eval/eval_behavior_agent.py` without `--navigation-only`
allows both `navigate_vln` and champion `vla_execute`. The Agent decides which
skill to use and when to hand off. It can also let VLA do its own navigation.
There is no navigation/manipulation control-mode switch, base lock, automatic
posture change, or prescribed skill sequence. All 23 VLA action dimensions
are forwarded unchanged. See `scripts/Behavior_eval/README.md` for complete
service and evaluation commands.

After local motion the gateway invalidates only stale action caches and
rolling-inpainting seeds before VLA resumes. It preserves the current stage,
voting history, task ID, counters and base-policy state. Consecutive VLA calls
retain normal caching. Only official episode resets/task changes restart the
stage as in the baseline. The updated champion server must be restarted and
advertise this capability; unsupported servers are rejected at handoff,
without a full-reset fallback. External movement and fresh action planning
remain differences from a pure baseline. Results record navigation calls,
`upstream_cache_invalidations_sent`, and `evaluator_resets_forwarded` separately.

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

## Continuous-frame Agent shadow preview

Once the champion policy on port 8011 and the LightNav server on port 8051 are
healthy, check the passive experiment without starting an episode:

```bash
cd /data/yxy/EMERGE-Humanoid
/home/embodied/miniconda3/envs/phyagentos/bin/python \
  scripts/Behavior_eval/run_lightnav_shadow.py --preflight-only
```

Then run instance index 0 of `turning_on_radio`:

```bash
/home/embodied/miniconda3/envs/phyagentos/bin/python \
  scripts/Behavior_eval/run_lightnav_shadow.py \
  --task-name turning_on_radio --instance-index 0 \
  --agent-model gpt-6.1-sol --headless --write-video
```

The default output directory is
`/data/yxy/outputs/Emerge/turning_on_radio/lightnav_shadow_301`. The official
rollout has its ordinary `json/` and `videos/` artifacts. The Agent's decision
and tool calls are in `agent_run/events.jsonl`, a compact summary is in
`lightnav_shadow_summary.json`, and every previewed frame and first waypoint
is saved under `workspace/artifacts/lightnav_preview/`.

The evaluator still receives only champion actions through a passive gateway.
The Agent can decide not to preview navigation at all; a zero-call episode is
valid evidence about its choice, not a LightNav failure. When it calls
`preview_navigation`, the tool reuses one LightNav session across fresh head
frames, resets at a new episode, and has no simulator action path. Its
`stop=true` and waypoint outputs are diagnostic suggestions, not arrival or
official success signals. This experiment does not measure closed-loop
navigation performance.

The preview currently reuses the tested WebSocket protocol client from the
G1/SONIC integration; no SONIC action adapter is imported or used. A common
protocol module can replace that import when both robot adapters are ready.

## Active LightNav to champion handoff ablation

Start the champion OpenPI service on port 8011, SAM3 on port 8002, and the
LightNav `vln` service on port 8051. Check connectivity without starting Isaac:

```bash
cd /data/yxy/EMERGE-Humanoid
/home/embodied/miniconda3/envs/phyagentos/bin/python \
  scripts/Behavior_eval/run_lightnav_handoff.py --preflight-only
```

Run the first public radio instance with official video enabled:

```bash
/home/embodied/miniconda3/envs/phyagentos/bin/python \
  scripts/Behavior_eval/run_lightnav_handoff.py \
  --task-name turning_on_radio --instance-index 0 \
  --nav-max-steps 900 --headless --write-video
```

The default output is
`/data/yxy/outputs/Emerge/turning_on_radio/lightnav_handoff_301`.
The script refuses to overwrite an existing run; use `--output-dir` for a
repeat. The LightNav trace and each input frame are saved under `workspace/`;
`navigation.mp4` is also rendered from those frames when ffmpeg is available,
even if the episode ends early. The SAM3 radio overlay is
`arrival_radio_overlay.png`, and the compact outcome
is `lightnav_handoff_summary.json`. A successful handoff also gets the
official evaluator's ordinary `json/` and `videos/` files.

The fixed runner uses a controlled gateway, not an EMERGE LLM Agent. It sends
one fresh head frame per navigation control period, converts the first model
waypoint to a bounded R1Pro base velocity, executes 7 or 8 official actions,
and replans. The R1Pro adapter preserves the other 20 action dimensions at
their observed hold pose. Only a LightNav stop candidate followed by a
separately segmented RGB-D radio within `--handoff-distance-m` permits the
champion policy to take over. If that gate fails, the runner does not let the
champion complete the missing navigation; the partial trace remains available
but an official final JSON/video may not be produced.

The proximity gate is an experimental criterion, **not** proof of arm reach or
official task success. Local navigation invalidates cached actions but preserves
the champion's stage state; handoff must still be judged from the trace and
the official result. The 30 Hz evaluator/4 Hz LightNav cadence and the
normalized base velocity mapping require real-simulator calibration before
large-scale runs. This ablation does not establish that an autonomous EMERGE
Agent would choose the same navigation skill or handoff time. Do not reuse the
SONIC movement-command adapter for R1Pro.

## Agent-guided navigation and body observation

Use the existing `eval_behavior_agent.py` entrypoint for an Agent-controlled
navigation-only debug run. The Agent reads an episode-specific `EMBODIED.md`
profile and the built-in `lightnav` skill, then chooses `navigate_vln` and/or `adjust_posture`
through `execute_robot_action`. The normal Controller, R1Pro driver, and
controlled BEHAVIOR gateway execute the actions. VLA, gripper, and object
manipulation actions remain disabled. The automatic visual monitor is disabled;
the main Agent may choose the existing `task_verification` or `object_location`
specialists itself. There is no fixed action/subagent order or radio-specific
arrival gate. Object localization requires a healthy configured SAM3 service;
it is optional at startup and checked/reported by preflight. Visual verification
uses the existing multi-camera images and does not require SAM3.

`navigate_vln` can be called repeatedly with different instructions and budgets.
The driver retains the WebSocket session and visual history in one episode,
resetting it on connection failure or an episode change. A call budget or
model stop returns control to the Agent; it does not terminate its mission.
All calls append the same default trace and head-frame sequence.

`adjust_posture` accepts a map of relative `joint_deltas_rad` for URDF torso
and arm joints. The head camera is fixed to the torso, and wrist cameras to
the grippers; there is no independent camera pan/tilt command. Each delta is
bounded to ±0.25 rad and interpolated at at most 0.01 rad per step with 10
settling steps. The base command stays zero and grippers/unspecified joints
hold their measured positions. Posture results include synchronized observations,
actual joint positions, link origins, and target-tracking error. URDF limits
and small changes do **not** provide collision avoidance, body-volume checks,
or contact sensing. The Agent chooses observation motions from available
evidence; the driver never inserts a default body posture.

`--max-steps` is the **shared episode action budget** for navigation and
posture. Each call's `max_steps` must fit `remaining_action_steps` in
ROBOT_STATE.md; each posture call must also fit the primitive limit.

The current gateway still needs the champion OpenPI service for its BEHAVIOR
websocket handshake, but `navigate_vln` executes local R1Pro actions and does
not ask the champion model for actions. If OpenPI is not already running, start
only that service in one terminal (not SAM3):

```bash
cd /data/yxy/EMERGE-Humanoid
BEHAVIOR_SERVICES=openpi bash scripts/Behavior_eval/start_behavior_services.sh
```

Start `lightnav-serve` with the command above in another terminal. Run the
visible experiment from a third terminal:

```bash
cd /data/yxy/EMERGE-Humanoid
/home/embodied/miniconda3/envs/phyagentos/bin/python scripts/Behavior_eval/eval_behavior_agent.py \
  --task-name turning_on_radio \
  --instance-index 1 \
  --navigation-only \
  --navigation-goal "Find the red-and-white radio on the coffee table in the living room and approach it." \
  --max-steps 900 \
  --agent-model gpt-6.1-sol \
  --episode-timeout 3600 \
  --no-headless \
  --show-agent-events \
  --write-video \
  --output-dir /data/yxy/outputs/Emerge/turning_on_radio/navigation_only_302_agent_body
```

The command above tests public index 1 (instance 302) and uses a fresh output
directory. Without `--output-dir`, the directory is
`/data/yxy/outputs/Emerge/<task>/navigation_only_<instance_id>`. The live
Isaac Sim window remains open for inspection after the Agent finishes until
Enter is pressed in the runner terminal. `workspace/lightnav_trace.jsonl`
records instructions, call IDs, each model frame, executed steps, and posture
end observations; `workspace/lightnav_frames/` contains all saved head images.
`agent_episode_result.json` includes `navigation_calls`, `posture_calls`,
total navigation steps in `navigation.nav_steps`, `posture_steps`, and
`total_robot_action_steps`. Main-Agent decisions and subagent calls remain in
the ordinary Agent logs/events. A model stop is **not** verified arrival, and
this run does **not** measure success at turning on the radio. On exit, the
runner sends the evaluator SIGINT and waits for its video writer to close
before process cleanup. With `--write-video` (the default), it also renders
`navigation_head.mp4` from all saved head frames at a fixed 4 fps. Posture-end
and action-boundary images are included; this supplementary clip is not a
uniform simulator-time recording. The official 30 Hz multi-camera rollout
includes actual posture motion. The supplementary clip remains reviewable if the
evaluator cannot finish its own MP4. The result reports whether graceful
shutdown succeeded and where the navigation clip was saved. Use a new
`--output-dir` for repeat runs.

Offline regression checks (no model requests or simulator):

```bash
/home/embodied/miniconda3/envs/phyagentos/bin/python -m unittest \
  scripts.Behavior_eval.checks.test_r1pro_kinematics \
  scripts.Behavior_eval.checks.test_r1pro_motion \
  scripts.Behavior_eval.checks.test_driver \
  scripts.Behavior_eval.checks.test_runner \
  integrations.lightnav_behavior.checks.test_active_navigation
```
