# LightNav-0 + SONIC 接入 EMERGE-Policy 的详细方案

## 1. 目标与边界

本方案把已经验证的：

```text
G1 head_camera + language
  → LightNav-0
  → local SE(2) first waypoint
  → LightNav2SonicAdapter
  → SONIC planner command
  → SONIC Kinematic Planner + WBC
  → G1 locomotion
```

封装成 EMERGE 可规划、可取消、可验证、可恢复的 Operational Skill。

第一阶段不做：

- GR00T manipulation；
- Cosmos/Imagination Skill；
- 完整地图或 SLAM；
- 修改 LightNav policy、SONIC planner 或 WBC；
- 将完整 waypoint sequence 直接送给 SONIC；
- 多机器人/真机部署。

## 2. 目标数据链路

```text
用户任务
  ↓
EMERGE Main Agent
  ├─ update_plan：记录导航 subgoal 与 done criterion
  └─ execute_robot_action(action_type="navigate_vln", ...)
       ↓
workspace/ACTION.md
       ↓
EMERGE Controller
       ↓
SonicLightNavDriver.execute_action(...)
  ├─ 订阅 head_camera / ego_view (:5555)
  ├─ 订阅 SONIC g1_debug yaw (:5557)
  ├─ 调用 LightNav WebSocket (:8051)
  ├─ first waypoint → LightNav2SonicAdapter
  ├─ 发布 SONIC planner packet (:5556)
  ├─ 检查 cancel/timeout/stop/error
  └─ 写最新图片、nav_state、执行结果
       ↓
SONIC Kinematic Planner → WBC → MuJoCo G1
       ↓
ACTION.md result + ROBOT_STATE.md + observation manifest
       ↓
Main Agent 验证、推进 PLAN 或压入 recovery branch
```

## 3. 为什么使用 external driver

SONIC 接入点应实现为 `BaseDriver` external plugin，原因是：

1. `execute_robot_action` 已经提供动作排队、单动作互斥、等待结果和取消协议。
2. Controller 已经负责 claim action、调用 driver、更新 `ACTION.md`/`ROBOT_STATE.md`。
3. `robot/plugins.py` 已支持 manifest 注册，不需要修改内置 driver registry。
4. Driver 可以在 `finally` 中保证发送 SONIC locomotion `IDLE`。
5. Main Agent 与低层 4 Hz 控制循环隔离，避免每个 waypoint 都经过 LLM。

首版无需修改 `Emerge/agent/tools/embodied.py`。尽管它的文字说明暂时只列 manipulation action，但 `action_type` schema 是自由字符串；driver profile 和 workspace skill 可以明确增加 `navigate_vln`。后续若扩展到多个 humanoid skill，再把 action capability schema 改为由 driver/profile 动态提供。

## 4. 建议新增的文件

实际实现时只在 fork 中新增独立目录：

```text
EMERGE-Humanoid/
├── integrations/lightnav_sonic_emerge/
│   ├── __init__.py
│   ├── Emerge_plugin.toml
│   ├── driver.py
│   ├── navigation_backend.py
│   ├── observation_writer.py
│   ├── enqueue_navigation.py
│   ├── profile_lightnav_sonic_g1.md
│   ├── driver_procthor.json
│   ├── README.md
│   ├── workspace_skill/
│   │   └── SKILL.md
│   └── checks/
│       ├── test_driver.py
│       ├── test_navigation_backend.py
│       ├── test_observation_writer.py
│       ├── test_enqueue_navigation.py
│       └── test_plugin.py
```

实际落地将测试放在 integration 内部，是因为仓库现有 `.gitignore` 会忽略根目录的整个
`tests`（包括同名子目录）；因此使用 `checks/` 目录，无需修改仓库级忽略规则，测试也可以
正常纳入版本管理。

不修改：

- `/home/embodied/LightNav-0/src/lightnav/`；
- `/home/embodied/GR00T-WholeBodyControl/gear_sonic*`；
- SONIC policy/planner 模型；
- LightNav checkpoint。

## 5. Plugin 与 Driver 契约

### 5.1 `Emerge_plugin.toml`

建议 manifest：

```toml
[plugin]
name = "lightnav-sonic-g1"
version = "0.1.0"

[driver]
name = "lightnav_sonic_g1"
module = "driver"
class = "SonicLightNavDriver"
profile_path = "profile_lightnav_sonic_g1.md"

[python]
sys_paths = ["."]
```

### 5.2 Driver 初始化参数

`driver_procthor.json` 建议包含：

```json
{
  "workspace": "/home/embodied/.Emerge/workspace",
  "lightnav_server": "ws://127.0.0.1:8051",
  "camera_endpoint": "tcp://127.0.0.1:5555",
  "camera_name": "ego_view",
  "sonic_feedback_endpoint": "tcp://127.0.0.1:5557",
  "sonic_feedback_topic": "g1_debug",
  "sonic_bind": "tcp://*:5556",
  "control_hz": 4.0,
  "input_timeout_s": 0.75,
  "default_max_steps": 120,
  "elastic_band_release_file": "/tmp/lightnav_sonic_release",
  "publish_observations": true
}
```

外部 driver 不会被 `controller.py` 自动注入 workspace（当前代码只对 `libero_mujoco` 自动注入），因此配置中必须显式给出 `workspace`。

### 5.3 支持的 action

第一阶段只需要一个主 action：

```json
{
  "action_type": "navigate_vln",
  "parameters": {
    "instruction": "请移动到桌子旁边，并在桌子旁边停下",
    "max_steps": 120,
    "control_hz": 4.0
  }
}
```

参数语义：

| 参数 | 必需 | 含义 |
|---|---:|---|
| `instruction` | 是 | 只描述当前导航 subgoal，不包含后续 manipulation |
| `max_steps` | 否 | bridge 控制周期上限；120/4 Hz = 30 秒，不是距离 |
| `control_hz` | 否 | 默认 4 Hz；同时决定 adapter 的 `dt` |

首版不暴露 `reset_session` 参数：每个 `navigate_vln` action 都创建新的 WebSocket client，
连接时自动执行 `login + reset`，同一 action 内维持连续 RGB history。

后续可增加恢复 primitive：

- `scan_environment`：按若干离散 yaw 角度旋转并保存观察；
- `turn_in_place`：明确旋转给定角度；
- `navigation_stop`：主动发 locomotion IDLE。

`scan_environment` 不应伪装成 LightNav 推理结果，它是 SONIC 上的显式 motion primitive，用于补足当前 LightNav 缺乏主动环视的问题。

## 6. `navigation_backend.py` 的职责

该模块复用随 integration 一起提交的 bridge 组件，因此其他开发者只需克隆本分支，无需复制本机 LightNav 仓库中的额外脚本：

```python
from .bridge.adapter import AdapterLimits, LightNav2SonicAdapter
from .bridge.lightnav_ws import LightNavWsClient
from .bridge.runner import SonicCameraSubscriber, SonicYawSubscriber
from .bridge.sonic_zmq import SonicZmqPublisher
```

它应提供一个有边界的同步执行方法，例如：

```python
run_navigation(
    instruction: str,
    max_steps: int,
    control_hz: float,
    cancel_check: Callable[[], str | None],
) -> NavigationResult
```

每个 tick：

1. 读取最新 JPEG 与 yaw，超过 `input_timeout_s` 即发送 IDLE。
2. 将 JPEG 和当前 instruction 送到 LightNav。
3. 严格解析 `stop` 与 waypoint shape/finite values。
4. 只取第一个 `[forward_m, lateral_m, delta_yaw_rad]`。
5. 用已验证的 adapter 转成 world-frame movement/facing/speed。
6. 发布标准 SONIC planner packet。
7. 保存最新观察和统计信息。
8. 调用 `cancel_check()`；收到原因立即 IDLE 并返回 `cancelled`。

所有退出路径都必须在 `finally` 中：

- 发送 locomotion `IDLE`；
- 关闭 LightNav WebSocket；
- 关闭 camera/yaw SUB；
- 关闭 SONIC PUB；
- 绝不发送 SONIC process-level `command.stop`。

### 6.1 完成与失败结果

建议 `NavigationResult` 至少包含：

```json
{
  "status": "completed|budget_exhausted|cancelled|failed",
  "reason": "lightnav_stop|max_steps|visual_monitor|stale_input|exception",
  "ticks": 37,
  "command_steps": 35,
  "elapsed_s": 9.25,
  "last_waypoint": [0.18, -0.02, 0.04],
  "last_yaw_rad": 1.42,
  "lightnav_stop": true
}
```

注意：`LightNav stop=true` 只能说明 policy 判定应停止，不等于 EMERGE 的语义任务已被外部证据验证。Driver action 可以结束，但 Main Agent 仍应根据 done criterion 判断是否推进 PLAN。

## 7. `driver.py` 的职责

### 7.1 `get_profile_path()`

返回 `profile_lightnav_sonic_g1.md`。Controller 启动后会复制为 workspace 的 `EMBODIED.md`。

Profile 应声明：

- 机器人是 Unitree G1 + SONIC locomotion；
- camera 为原始 `head_camera`，网络流名为 `ego_view`；
- 支持 `navigate_vln`；
- 速度、控制频率和超时限制；
- 一次只允许一个 `:5556` publisher；
- 目标是否到达必须结合观察验证；
- 首版禁止主 Agent 调用 LIBERO 专用 object_location 获取不存在的标定坐标。

### 7.2 `load_scene()`

SONIC ProcTHOR 场景由外部 `room_sim` 启动，因此首版可为 no-op，但应检查 camera、feedback、LightNav server 和 SONIC subscriber 的连接状态，并把结果写入 runtime state。

### 7.3 `execute_action()`

处理 `navigate_vln`，调用 backend 并把结果序列化为清晰字符串。未知 action 返回 `Unknown action type`，让 Controller 标记 failed。

### 7.4 `get_runtime_state()`

建议回写：

```json
{
  "robots": {
    "g1_sonic_001": {
      "connection_state": {
        "status": "connected",
        "lightnav_server": "ws://127.0.0.1:8051",
        "camera_endpoint": "tcp://127.0.0.1:5555",
        "sonic_feedback_endpoint": "tcp://127.0.0.1:5557"
      },
      "robot_pose": {
        "frame": "sonic_world",
        "yaw": 1.42
      },
      "nav_state": {
        "mode": "navigating",
        "status": "running",
        "instruction": "move to the table",
        "ticks": 21,
        "command_steps": 20,
        "max_steps": 120,
        "last_waypoint": [0.18, 0.01, 0.03],
        "last_error": null
      }
    }
  }
}
```

当前 `g1_debug` 的可靠字段是 `base_quat`，已有 bridge 只据此得到 yaw。其 `base_trans_measured` 在官方注释中是固定默认值，所以第一阶段不要伪造 x/y 全局位姿或 path progress。后续若仿真侧增加真实 base pose 发布，再扩充 `robot_pose.x/y`。

## 8. Observation 与验证

### 8.1 第一阶段

Driver 每个 tick 将最新 JPEG 原子保存为：

```text
~/.Emerge/workspace/artifacts/observations/ego_view.png
```

并原子覆盖：

```text
~/.Emerge/workspace/artifacts/observations/observation.json
```

最小内容：

```json
{
  "revision": 42,
  "reference_view": "ego_view",
  "views": [
    {
      "name": "ego_view",
      "image_path": "artifacts/observations/ego_view.png"
    }
  ]
}
```

这足够 Task Verification 和 Visual Monitor 查看第一视角，但不够 VGGT/SAM3 object localization。

### 8.2 第二阶段

获得准确 head camera intrinsics 和 `T_world_camera` 后，再把标定写入 manifest；必要时加入多个固定第三视角。之后才启用 Object Location 和更可靠的导航验证。

### 8.3 第一阶段完成判断

按可靠性从高到低组合：

1. Visual Monitor / Task Verification 明确看到目标且满足距离/停靠关系；
2. LightNav `stop=true`；
3. 控制预算耗尽；
4. stale camera/yaw、WebSocket 或 ZMQ 异常。

其中 2 只能触发动作结束和验证，不能单独宣告整项任务成功；3/4 应触发恢复或报告失败。

## 9. Navigation workspace skill

`workspace_skill/SKILL.md` 应指导 Main Agent：

1. 对多步任务先建立 `PLAN.md`。
2. `navigate_vln` instruction 只包含当前导航目标，例如“移动到桌子旁并停下”。
3. 不把后续抓取、放置等操作混进 LightNav instruction。
4. `max_steps` 是时间预算，不是米数。
5. action completed 不等于 subgoal achieved；读取最新 `ROBOT_STATE.md` 并验证。
6. 目标不可见、无进展或方向错误时，优先压入扫描/转向 recovery branch，而不是无限重复同一指令。
7. 两次相同策略无进展后必须换朝向或重规划。

## 10. 安全与并发约束

- EMERGE Controller driver 是 `tcp://*:5556` 的唯一 PUB owner。
- Controller 运行期间不要启动 integration 自带的 standalone `bridge.runner` 或 `bridge.teleop`。
- 启动时先用 Elastic Band 托住 G1；收到有效 yaw 后由 driver 创建 release file。
- 相机、yaw、LightNav 回复或 ZMQ 任一 stale/error 都发送 IDLE。
- `cancel_check` 至少每个 4 Hz tick 检查一次。
- `close()` 和异常 `finally` 必须发送 IDLE。
- 第一阶段只在 MuJoCo 中验证，不上真机。

## 11. 实施顺序与测试标准

### Phase A：纯单元测试

- manifest 可以注册和加载 external driver；
- action 参数校验；
- LightNav stop、无 waypoint、NaN、stale camera/yaw fallback；
- cancel 后一定发送 IDLE；
- observation 图片/manifest 原子更新；
- runtime state 不虚构 x/y；
- unknown action 正确失败。

### Phase B：Mock integration

使用固定 waypoint 但走完整 EMERGE 路径：

```text
Main Agent/headless request
→ ACTION.md
→ Controller driver
→ adapter
→ SONIC planner
→ G1 locomotion
→ ACTION result/ROBOT_STATE
```

### Phase C：Real LightNav integration

本机 GPU 启动 LightNav server，输入真实 head camera 和语言，确认终端日志与产物中出现真实 waypoint。

### Phase D：Agent recovery

目标初始位于侧方或后方，验证 Main Agent 能在失败后调用 scan/turn recovery，再次执行导航；这一步才接近“主动探索”，不是第一阶段连接成功的必要条件。

## 12. 实现完成后的终端命令

以下命令是 driver 文件实现后的运行方式。实现前不要执行 plugin 注册和 Controller 命令。

### 12.1 安装 workspace skill

```bash
mkdir -p /home/embodied/.Emerge/workspace/skills/lightnav-sonic-navigation

cp /home/embodied/EMERGE-Humanoid/integrations/lightnav_sonic_emerge/workspace_skill/SKILL.md \
  /home/embodied/.Emerge/workspace/skills/lightnav-sonic-navigation/SKILL.md
```

### 12.2 注册 external driver

```bash
cd /home/embodied/EMERGE-Humanoid

conda run -n EmergePolicy python -c \
  "from robot.plugins import register_plugin; print(register_plugin('/home/embodied/EMERGE-Humanoid/integrations/lightnav_sonic_emerge'))"
```

检查：

```bash
conda run -n EmergePolicy python -c \
  "from robot.drivers import list_drivers; print(list_drivers())"
```

输出应包含：

```text
lightnav_sonic_g1
```

### 12.3 终端 A：ProcTHOR + G1 MuJoCo

每次完整重启仿真前清理旧的临时 release signal：

```bash
rm -f /tmp/lightnav_sonic_release

cd /home/embodied/EMERGE-Humanoid
source /home/embodied/GR00T-WholeBodyControl/.venv_sim/bin/activate

PYTHONPATH=. python -m integrations.lightnav_sonic_emerge.bridge.room_sim \
  --sonic-root /home/embodied/GR00T-WholeBodyControl \
  --lightnav-root /home/embodied/LightNav-0 \
  --environment procthor \
  --run \
  --gui \
  --publish-ego \
  --camera-port 5555 \
  --elastic-band-release-file /tmp/lightnav_sonic_release
```

### 12.4 可选终端 A2：相机查看器

```bash
cd /home/embodied/GR00T-WholeBodyControl
source .venv_sim/bin/activate

python gear_sonic/scripts/run_camera_viewer.py \
  --camera-host 127.0.0.1 \
  --camera-port 5555
```

### 12.5 终端 B：SONIC deployment

```bash
cd /home/embodied/GR00T-WholeBodyControl/gear_sonic_deploy
source scripts/setup_env.sh

./target/release/g1_deploy_onnx_ref \
  lo \
  policy/release/model_decoder.onnx \
  reference/example/ \
  --obs-config policy/release/observation_config.yaml \
  --encoder-file policy/release/model_encoder.onnx \
  --planner-file planner/target_vel/V2/planner_sonic.onnx \
  --input-type zmq_manager \
  --output-type all \
  --zmq-host 127.0.0.1 \
  --disable-crc-check
```

### 12.6 终端 C：LightNav 本地 GPU server

```bash
cd /home/embodied/LightNav-0

conda run -n lightnav-0 env CUDA_VISIBLE_DEVICES=0 PORT=8051 \
  lightnav-serve \
  --task vln \
  --model_path checkpoints/LightNav-0 \
  --backend vllm_local \
  --gpu_memory_utilization 0.50
```

### 12.7 终端 D：EMERGE Controller + SONIC navigation driver

这里由 driver bind `5556`，不要再启动旧 bridge runner。

```bash
cd /home/embodied/EMERGE-Humanoid
conda activate EmergePolicy

export PYTHONPATH=/home/embodied/EMERGE-Humanoid

python -m robot.controller \
  --driver lightnav_sonic_g1 \
  --workspace /home/embodied/.Emerge/workspace \
  --driver-config integrations/lightnav_sonic_emerge/driver_procthor.json \
  --interval 0.1
```

### 12.8 终端 E：EMERGE Main Agent

交互模式：

```bash
cd /home/embodied/EMERGE-Humanoid
conda activate EmergePolicy

emerge \
  --workspace /home/embodied/.Emerge/workspace
```

输入示例：

```text
请移动到桌子旁边，并在桌子旁边停下。先建立计划，使用 LightNav-0 + SONIC 导航能力；动作结束后根据最新观察验证是否真正到达。
```

自动化单次任务：

```bash
cd /home/embodied/EMERGE-Humanoid
conda activate EmergePolicy

python -m Emerge.cli.headless \
  "请移动到桌子旁边，并在桌子旁边停下；完成后验证结果。" \
  --workspace /home/embodied/.Emerge/workspace \
  --timeout-s 300 \
  --max-iterations 20 \
  --output-dir /home/embodied/EMERGE-Humanoid/outputs/lightnav_sonic_emerge_run
```

## 13. 验收标准

接入成功需要同时满足：

1. Main Agent 自主生成含导航 subgoal 的 `PLAN.md`。
2. `ACTION.md` 出现 `navigate_vln`，并完成 `pending → running → terminal`。
3. Driver 收到真实 `ego_view` 与 yaw。
4. LightNav server 返回真实 waypoint，而不是 mock。
5. Adapter 产生 SONIC 标准 planner command。
6. G1 在 ProcTHOR MuJoCo 中产生对应 locomotion。
7. 完成、取消、异常和进程退出都会发送 locomotion IDLE。
8. `ROBOT_STATE.md` 记录真实 nav state，不伪造位置。
9. Main Agent 不把“action completed”直接当成“已到达目标”。

第一阶段满足 1–8 即可认为导航 skill 已正确接入；第 9 项和主动 scan/recovery 的稳定性属于随后要重点验证的 agentic navigation 能力。
