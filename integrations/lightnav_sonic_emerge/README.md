# LightNav-0 + NVIDIA SONIC Navigation for EMERGE-Humanoid

This integration exposes visual-language navigation as the bounded EMERGE
action `navigate_vln`. LightNav-0 predicts a robot-local SE(2) waypoint from
the G1 head-camera image and a language instruction; the bridge converts that
waypoint into SONIC's existing planner command, and SONIC produces whole-body
G1 locomotion in MuJoCo.

The implementation is intentionally isolated from the upstream LightNav-0 and
SONIC source trees. It uses public process boundaries (WebSocket and ZMQ), does
not modify either policy, and ships its bridge implementation inside this
directory so a clone of this branch contains all integration code.

> **Research status:** the deterministic queue path has been tested end to end
> with the real LightNav-0 checkpoint on an RTX 4090 and a SONIC-controlled G1
> in the ProcTHOR MuJoCo room. In one recorded run, the table instruction ended
> after 50 ticks / 49 planner commands with `LightNav stop=true`, and the G1
> reached the table. This is a functional prototype, not a navigation success
> guarantee or a real-robot release. Main-Agent planning has not yet been
> evaluated end to end.

## System architecture

```text
manual enqueue tool OR EMERGE Main Agent (not yet evaluated)
                         |
                         v
                 workspace/ACTION.md
                         |
                         v
 EMERGE Controller -> SonicLightNavDriver.navigate_vln
                         |
      +------------------+------------------+
      |                                     |
      v                                     v
G1 head_camera / ego_view :5555       G1 world yaw :5557
      |                                     |
      +------------------+------------------+
                         |
                         v
          LightNav-0 WebSocket server :8051
        RGB + instruction -> [forward, lateral, yaw]
                         |
                         v
              LightNav2SonicAdapter
       local waypoint -> world direction/heading/speed
                         |
                         v
              SONIC planner ZMQ :5556
                         |
                         v
       Kinematic Planner -> WBC -> G1 locomotion
```

The Controller is the only owner of the SONIC planner publisher while an
action is running. Do not run the standalone bridge runner or keyboard teleop
at the same time, because only one process may bind `tcp://*:5556`.

## What this branch adds

- An external EMERGE driver named `lightnav_sonic_g1`.
- A bounded and cancellable `navigate_vln` action.
- A vendored waypoint adapter and LightNav/SONIC transport layer.
- Safety handling for stale input, invalid model output, exceptions,
  cancellation, action budgets, and final locomotion `IDLE`.
- Ego-image publication into the EMERGE observation workspace.
- A deterministic action-enqueue utility that tests the real model/control
  path without requiring an LLM planner.
- A G1 + ProcTHOR scene composer and simulator launcher. Derived scenes are
  generated at runtime; official LightNav and SONIC assets are not copied.
- Unit and protocol checks for the driver, action queue, adapter, WebSocket
  parser, ZMQ packet, feedback parsing, and fallback behavior.

## Repository layout

```text
integrations/lightnav_sonic_emerge/
├── Emerge_plugin.toml          # external-driver manifest
├── driver.py                   # EMERGE action and runtime state
├── navigation_backend.py       # bounded closed-loop action lifecycle
├── observation_writer.py       # workspace ego observations
├── enqueue_navigation.py       # deterministic ACTION.md producer
├── driver_procthor.json        # portable local endpoint defaults
├── profile_lightnav_sonic_g1.md
├── workspace_skill/SKILL.md    # Main-Agent navigation guidance
├── bridge/                     # adapter, protocols and room simulator
└── checks/                     # offline unit/protocol tests
```

Detailed Chinese documents are available in
[`apply_docs/`](../../apply_docs/), especially the
[execution guide](../../apply_docs/04_导航Skill执行与测试终端指令.md) and the
[implementation inventory](../../apply_docs/05_导航Skill实现文件与改动说明.md).

## Tested revisions and host

The integration is intentionally revision-tolerant, but the current result was
obtained with the following setup:

| Component | Tested value |
|---|---|
| EMERGE-Humanoid | branch `humanoid-dev` |
| LightNav-0 | `lightorigins/LightNav-0` at `0e9971784a04` |
| SONIC | `NVlabs/GR00T-WholeBodyControl`, base checkout `542842820d35` |
| GPU | NVIDIA GeForce RTX 4090, 24 GB |
| OS | Ubuntu 22.04 |
| EMERGE Python | 3.12 |
| LightNav Python | 3.11 |
| SONIC MuJoCo Python | repository `.venv_sim` (3.10 in the tested setup) |

Newer upstream revisions may require small compatibility updates. Pin these
commits when exact reproduction matters.

## Fresh-clone setup

The commands below assume three sibling checkouts. Replace the workspace path
if needed; no code depends on `/home/embodied`.

```bash
export HUMANOID_WS="$HOME/humanoid-navigation"
mkdir -p "$HUMANOID_WS"
cd "$HUMANOID_WS"

git clone --branch humanoid-dev \
  https://github.com/hrm-666/EMERGE-Humanoid.git
git clone https://github.com/lightorigins/LightNav-0.git
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git

export EMERGE_ROOT="$HUMANOID_WS/EMERGE-Humanoid"
export LIGHTNAV_ROOT="$HUMANOID_WS/LightNav-0"
export SONIC_ROOT="$HUMANOID_WS/GR00T-WholeBodyControl"
```

### 1. Install EMERGE-Humanoid

For navigation-only work, the full LIBERO/OpenPI/Cosmos stack is unnecessary.
Install the main package and the integration dependencies:

```bash
conda create -n EmergePolicy python=3.12 -y
conda activate EmergePolicy
python -m pip install --upgrade pip wheel "setuptools<82"
python -m pip install -e "$EMERGE_ROOT"
python -m pip install -r \
  "$EMERGE_ROOT/integrations/lightnav_sonic_emerge/bridge/requirements.txt"
```

If pip reports `Missing dependencies for SOCKS support`, the shell has a SOCKS
proxy configured but the isolated build environment lacks SOCKS support. Either
install `PySocks` first or disable proxy variables for this one command:

```bash
env -u ALL_PROXY -u all_proxy \
    -u HTTP_PROXY -u http_proxy \
    -u HTTPS_PROXY -u https_proxy \
  python -m pip install -e "$EMERGE_ROOT"
```

Initialize EMERGE and register the external driver once:

```bash
emerge onboard

python -c "from robot.plugins import register_plugin; \
s=register_plugin('${EMERGE_ROOT}/integrations/lightnav_sonic_emerge'); \
print(s.driver_name, s.dotted_path)"

mkdir -p "$HOME/.Emerge/workspace/skills/lightnav-sonic-navigation"
cp "$EMERGE_ROOT/integrations/lightnav_sonic_emerge/workspace_skill/SKILL.md" \
  "$HOME/.Emerge/workspace/skills/lightnav-sonic-navigation/SKILL.md"
```

Expected registration output includes:

```text
lightnav_sonic_g1 driver.SonicLightNavDriver
```

### 2. Install LightNav-0 and download the checkpoint

Follow the upstream LightNav instructions if they change. The tested local-vLLM
setup was:

```bash
conda create -n lightnav-0 python=3.11 -y
conda activate lightnav-0
python -m pip install --upgrade pip
python -m pip install -e "${LIGHTNAV_ROOT}[vllm,video]"

hf download LightOriginsHQ/LightNav-0 \
  --local-dir "$LIGHTNAV_ROOT/checkpoints/LightNav-0"
```

The WebSocket endpoint used below is local IPC. Images and prompts are inferred
by the checkpoint on the local GPU; they are not sent to a cloud inference API.

### 3. Install SONIC simulation/deployment

Use the official SONIC installation instructions for the checked-out revision,
including Git LFS assets, the MuJoCo simulation environment, policy/model
downloads, and the deployment build. Verify the stock SONIC Sim2Sim example
before adding LightNav.

This repository deliberately does not copy SONIC weights, TensorRT engines, G1
assets, or generated build products. The runtime commands below expect:

- `$SONIC_ROOT/.venv_sim` to exist;
- `$SONIC_ROOT/gear_sonic_deploy/deploy.sh` to run successfully;
- the official G1 `scene_43dof.xml` and WBC assets to be available.

## Offline verification

No GPU, checkpoint, MuJoCo process, or SONIC deployment is required for these
tests:

```bash
cd "$EMERGE_ROOT"
conda activate EmergePolicy
export PYTHONPATH="$EMERGE_ROOT"

python -m compileall -q integrations/lightnav_sonic_emerge
python -m unittest discover \
  -s integrations/lightnav_sonic_emerge/checks -v
```

Expected result: **41 tests pass**.

## End-to-end ProcTHOR run

Use one terminal per process. The order matters. Before starting a new MuJoCo
episode, after all old processes have exited, remove the stale one-episode
safety signal:

```bash
rm -f /tmp/lightnav_sonic_release
```

### Terminal A — ProcTHOR room and G1 MuJoCo simulation

```bash
cd "$EMERGE_ROOT"
source "$SONIC_ROOT/.venv_sim/bin/activate"
PYTHONPATH="$EMERGE_ROOT" \
python -m integrations.lightnav_sonic_emerge.bridge.room_sim \
  --sonic-root "$SONIC_ROOT" \
  --lightnav-root "$LIGHTNAV_ROOT" \
  --environment procthor \
  --output-scene /tmp/lightnav_sonic_room/scene_43dof_lightnav_procthor.xml \
  --run \
  --gui \
  --publish-ego \
  --camera-port 5555 \
  --elastic-band-release-file /tmp/lightnav_sonic_release
```

`--gui` keeps the third-person view. `--publish-ego` renders the original G1
`head_camera` and publishes it as `ego_view`. The room is derived from
LightNav-0's official `mujoco_demo` ProcTHOR asset and SONIC's official G1 scene.
The generated MJCF lives in `/tmp`; neither upstream repository is edited.

### Terminal B — SONIC deployment

```bash
cd "$SONIC_ROOT/gear_sonic_deploy"
bash deploy.sh \
  --input-type zmq_manager \
  --zmq-host 127.0.0.1 \
  sim
```

This selects SONIC's existing ZMQ planner input and the simulation backend.

### Terminal C — optional ego-camera viewer

```bash
cd "$SONIC_ROOT"
source .venv_sim/bin/activate
python gear_sonic/scripts/run_camera_viewer.py \
  --camera-host 127.0.0.1 \
  --camera-port 5555
```

### Terminal D — real LightNav-0 inference

```bash
cd "$LIGHTNAV_ROOT"
conda activate lightnav-0
CUDA_VISIBLE_DEVICES=0 PORT=8051 \
lightnav-serve \
  --task vln \
  --model_path checkpoints/LightNav-0 \
  --backend vllm_local \
  --gpu_memory_utilization 0.70 \
  --max_batch_size 1
```

The reduced vLLM memory fraction leaves headroom for SONIC and MuJoCo on one
24 GB GPU. If CUDA graph capture still runs out of memory, retry with
`VLN_VLLM_ENFORCE_EAGER=1` before `lightnav-serve`.

### Terminal E — EMERGE Controller

```bash
cd "$EMERGE_ROOT"
conda activate EmergePolicy
export PYTHONPATH="$EMERGE_ROOT"
python -m robot.controller \
  --driver lightnav_sonic_g1 \
  --workspace "$HOME/.Emerge/workspace" \
  --driver-config \
    "$EMERGE_ROOT/integrations/lightnav_sonic_emerge/driver_procthor.json" \
  --interval 0.25
```

Wait for `Watching ACTION.md ...` before submitting an action.

### Terminal F — deterministic real-model action

```bash
cd "$EMERGE_ROOT"
conda activate EmergePolicy
python -m integrations.lightnav_sonic_emerge.enqueue_navigation \
  --workspace "$HOME/.Emerge/workspace" \
  --instruction "Please move next to the table and stop" \
  --max-steps 120 \
  --control-hz 4
```

This is **not mock navigation**. It bypasses only Main-Agent planning; the
camera, LightNav checkpoint, adapter, SONIC planner/WBC, and G1 simulation are
all in the loop. A second action can be submitted after the previous one is no
longer `pending` or `running`, without restarting terminals A-E.

`max_steps` is a control-tick budget, not a distance. At 4 Hz, 120 ticks provide
about 30 seconds of nominal control time, plus model-inference latency.

Inspect results with:

```bash
sed -n '1,260p' "$HOME/.Emerge/workspace/ACTION.md"
sed -n '1,320p' "$HOME/.Emerge/workspace/ROBOT_STATE.md"
python -m json.tool \
  "$HOME/.Emerge/workspace/artifacts/observations/observation.json"
```

## Main-Agent use

After configuring an EMERGE-supported model provider, the installed workspace
skill instructs the Main Agent to call `navigate_vln`, inspect the resulting
observation, and avoid treating action completion as semantic success. For
example:

```bash
emerge \
  --workspace "$HOME/.Emerge/workspace" \
  --model <configured-provider/model>
```

Then request a bounded, observable task such as: “Move next to the table and
stop; verify arrival using the latest camera observation.”

This Main-Agent route is structurally connected but **has not yet been evaluated
in the current prototype**. Use the deterministic enqueue path first when
debugging the robotics stack.

## Action and transport contract

The action payload is:

```json
{
  "action_type": "navigate_vln",
  "parameters": {
    "instruction": "move next to the table and stop",
    "max_steps": 120,
    "control_hz": 4.0
  }
}
```

Default local endpoints:

| Endpoint | Producer | Consumer | Data |
|---|---|---|---|
| `ws://127.0.0.1:8051` | LightNav server | Driver | VLN request/reply |
| `tcp://127.0.0.1:5555` | MuJoCo | Driver/viewer | `ego_view` JPEG |
| `tcp://127.0.0.1:5557` | SONIC | Driver | `g1_debug` base quaternion |
| `tcp://*:5556` | Driver | SONIC | planner command packets |

The first LightNav waypoint is interpreted as
`[forward_m, lateral_left_m, delta_yaw_rad]` in the robot frame. The adapter
rotates translational velocity into SONIC's world frame, clips translation and
yaw rates, emits `SLOW_WALK`, and sends `IDLE` for near, invalid, stale, stopped,
cancelled, or failed inputs. A normal navigation stop never sends SONIC's
process-level shutdown command.

## Safety and operational notes

- Run only one planner publisher on port 5556.
- The action ceiling defaults to 120 ticks and rejects larger requests unless
  the driver configuration is deliberately changed.
- Every action exit path attempts to send SONIC locomotion `IDLE`.
- `LightNav stop=true` means the policy requested termination; it does not prove
  that the language goal is satisfied. Verify the latest observation.
- The Elastic Band release file is per MuJoCo episode. Delete it before a new
  simulator start, not between actions in the same episode.
- Stop the Controller before using the optional keyboard teleop.
- Keep simulation validation separate from real-robot deployment. The current
  work has not been safety-qualified for hardware.

## Known limitations

- The original G1 `head_camera` is pitched downward relative to the mostly
  level ego imagery used by LightNav-0. Large floor coverage causes a domain
  gap and reduces object-search performance.
- Navigation is much more reliable when the target is initially visible.
  LightNav-0 does not consistently perform an explicit panoramic search when a
  target starts behind the robot.
- The first version consumes only the first predicted waypoint and does not send
  a full trajectory to SONIC.
- Completion is not paired with an independent semantic-goal verifier.
- ProcTHOR composition currently targets the tested LightNav demo asset and
  SONIC 43-DoF G1 scene.
- Main-Agent recovery planning and a UniNav checkpoint are future work.

## Development rules

- Keep official LightNav-0 and SONIC sources unchanged.
- Put integration changes and tests in this directory.
- Do not commit checkpoints, engines, generated MJCF, observations, API keys, or
  local workspace state.
- Run the 41 offline checks before opening a pull request.
- Report the upstream revisions, GPU, instruction, tick budget, and whether a
  result used manual enqueue or a Main Agent.

## Upstream projects

- [EMERGE-Policy paper](https://arxiv.org/abs/2608.29896)
- [EMERGE-Policy repository](https://github.com/EMERGE-Policy/EMERGE-Policy)
- [LightNav-0](https://github.com/lightorigins/LightNav-0)
- [NVIDIA GR00T Whole-Body Control / SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl)

Please cite the upstream projects according to their own documentation and
retain their respective licenses when using their code, assets, or weights.
