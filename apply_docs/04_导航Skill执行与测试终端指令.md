# LightNav-0 + SONIC 导航 Skill：执行与测试终端指令

## 1. 当前实现对应的数据流

```text
EMERGE Main Agent 或手动入队工具
  → ~/.Emerge/workspace/ACTION.md
  → EMERGE Controller
  → SonicLightNavDriver.navigate_vln
  → head_camera / ego_view (:5555) + SONIC yaw (:5557)
  → LightNav-0 WebSocket (:8051)
  → first local waypoint
  → 既有 LightNav2SonicAdapter
  → SONIC planner packet (:5556)
  → SONIC Kinematic Planner + WBC
  → ProcTHOR 中的 G1 locomotion
```

下面命令使用：

- EMERGE 环境：`EmergePolicy`；
- LightNav 推理环境：`lightnav-0`；
- SONIC deployment 环境：由其 `deploy.sh` 管理；
- MuJoCo 环境：`GR00T-WholeBodyControl/.venv_sim`；
- G1 原始 `head_camera`，网络图像名为 `ego_view`；
- 已有 ProcTHOR 派生场景，不启动旧的 standalone bridge runner。

## 2. 只需执行一次的 EMERGE 初始化

### 2.1 初始化 workspace

```bash
cd /home/embodied/EMERGE-Humanoid
conda activate EmergePolicy
env -u ALL_PROXY -u all_proxy \
    -u HTTP_PROXY -u http_proxy \
    -u HTTPS_PROXY -u https_proxy \
    -u PYTHONPATH \
  python -m pip install -e .
emerge onboard
```

`emerge onboard` 创建默认配置 `~/.Emerge/config.json` 和 workspace
`~/.Emerge/workspace`。如果配置已经存在，选择保留并刷新现有配置即可，不要覆盖已有 API key。

这里仅对这一条 pip 命令取消代理变量。本机 shell 当前设置了
`ALL_PROXY=socks://127.0.0.1:7897/`，但 pip 的隔离构建环境没有 PySocks；直接运行
`python -m pip install -e .` 会在下载 `hatchling` 时报告
`Missing dependencies for SOCKS support`。上述命令不会修改全局代理设置，后续终端仍保持原配置。

### 2.2 注册 external driver

```bash
cd /home/embodied/EMERGE-Humanoid
conda activate EmergePolicy
python -c "from robot.plugins import register_plugin; s=register_plugin('/home/embodied/EMERGE-Humanoid/integrations/lightnav_sonic_emerge'); print(s.driver_name, s.dotted_path)"
```

预期输出包含：

```text
lightnav_sonic_g1 driver.SonicLightNavDriver
```

安装 Main Agent 的导航规则：

```bash
mkdir -p /home/embodied/.Emerge/workspace/skills/lightnav-sonic-navigation
cp /home/embodied/EMERGE-Humanoid/integrations/lightnav_sonic_emerge/workspace_skill/SKILL.md \
  /home/embodied/.Emerge/workspace/skills/lightnav-sonic-navigation/SKILL.md
```

验证注册结果：

```bash
cd /home/embodied/EMERGE-Humanoid
conda activate EmergePolicy
python -c "from robot.drivers import list_drivers; print(list_drivers())"
emerge wsinit
```

`list_drivers()` 中应出现 `lightnav_sonic_g1`。

### 2.3 仅在要运行 Main Agent 时配置 LLM

确定性动作队列测试不需要 LLM。若使用 OpenAI Codex OAuth，可执行：

```bash
conda activate EmergePolicy
emerge provider login openai-codex
```

也可以继续使用 `~/.Emerge/config.json` 中已经配置的其他 provider。不要把 API key 写进本仓库。

## 3. 先运行离线测试

```bash
cd /home/embodied/EMERGE-Humanoid
conda activate EmergePolicy
export PYTHONPATH=/home/embodied/EMERGE-Humanoid
python -m compileall -q \
  integrations/lightnav_sonic_emerge \
  integrations/lightnav_sonic_emerge/checks
python -m unittest discover \
  -s integrations/lightnav_sonic_emerge/checks -v
```

预期为 41 项测试全部 `OK`。这些测试不启动 GPU、MuJoCo 或 SONIC，验证的是：

- external plugin 注册和异目录加载；
- Driver 参数及结果语义；
- 动作取消、最大步数、输入陈旧和异常安全停止；
- LightNav `stop=true`；
- 最终 SONIC `IDLE`；
- ego 图像及 observation manifest；
- 标准 `ACTION.md` 导航动作入队。

## 4. 每轮完整仿真开始前

第一次测试，或者上一次异常退出后 `ACTION.md` 中仍有 `pending/running` 动作时，先在所有
Controller/Agent 都已关闭的情况下重置当前 episode：

```bash
cd /home/embodied/EMERGE-Humanoid
conda activate EmergePolicy
python -c "from pathlib import Path; from Emerge.runtime.workspace import reset_workspace_context; reset_workspace_context(Path('/home/embodied/.Emerge/workspace'))"
```

该命令使用 EMERGE 自带的 workspace reset：清空旧 `ACTION/PLAN/ROBOT_STATE/EMBODIED`
任务状态和旧 observation，但保留 `skills/`、配置以及 `runs/` 历史。这样不会把 onboard
模板中的 Go2、苹果、杯子等示例状态带入 G1 导航。不要在 Controller 正在执行动作时运行。

所有旧进程已经关闭后，仅删除这一轮遗留的 Elastic Band 信号文件：

```bash
rm -f /tmp/lightnav_sonic_release
```

以下终端 A～F 要同时保留。`tcp://*:5556` 只能有一个发布者，因此不要同时运行：

- integration 自带的 standalone `bridge.runner`；
- integration 自带的 `bridge.teleop`；
- 另一个 EMERGE Controller。

## 5. 终端 A：启动 ProcTHOR + G1 MuJoCo

```bash
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

参数含义：

- `--lightnav-root`：定位官方 LightNav checkout 中的 `mujoco_demo` 场景资产；
- `--environment procthor`：使用 LightNav `mujoco_demo` 中的 ProcTHOR 房间资产；
- `--run`：完成场景合成和校验后进入仿真循环；
- `--gui`：保留全局第三人称 MuJoCo 窗口；
- `--publish-ego`：渲染 G1 原始 `head_camera`，并以 `ego_view` 发布；
- `--camera-port 5555`：发布 ego JPEG 的 ZMQ 端口；
- `--elastic-band-release-file`：在 Controller 真正收到相机和 yaw 前托住机器人。

看到 `waiting for release signal` 是正常状态。信号文件会由 Driver 在收到有效输入后创建。

## 6. 终端 B：启动 SONIC deployment

```bash
cd /home/embodied/GR00T-WholeBodyControl/gear_sonic_deploy
bash deploy.sh \
  --input-type zmq_manager \
  --zmq-host 127.0.0.1 \
  sim
```

参数含义：

- `--input-type zmq_manager`：让 SONIC 从现有 ZMQ planner 接口接收高层指令；
- `--zmq-host 127.0.0.1`：连接本机 Controller/Driver 在 `5556` 端口发布的 packet；
- `sim`：连接 MuJoCo，而不是真机。

## 7. 终端 C（建议）：显示 ego 图像

```bash
cd /home/embodied/GR00T-WholeBodyControl
source .venv_sim/bin/activate
python gear_sonic/scripts/run_camera_viewer.py \
  --camera-host 127.0.0.1 \
  --camera-port 5555
```

该窗口只显示实际送给 LightNav 的 `ego_view`；终端 A 的 MuJoCo GUI 继续显示第三人称视角。

## 8. 终端 D：启动真实 LightNav-0 GPU 推理

```bash
cd /home/embodied/LightNav-0
conda activate lightnav-0
CUDA_VISIBLE_DEVICES=0 PORT=8051 \
  lightnav-serve \
  --task vln \
  --model_path checkpoints/LightNav-0 \
  --backend vllm_local \
  --gpu_memory_utilization 0.70 \
  --max_batch_size 1
```

参数含义：

- `CUDA_VISIBLE_DEVICES=0`：使用第 0 张 GPU；
- `PORT=8051`：WebSocket server 监听端口；
- `--task vln`：启用视觉语言导航任务；
- `--model_path`：本地 LightNav checkpoint；
- `--backend vllm_local`：模型在本机 GPU 推理，WebSocket 只是本机进程间协议，不是云端推理。
- `--gpu_memory_utilization 0.70`：本测试需要同时运行 MuJoCo 和 SONIC，降低 vLLM 的启动显存
  门槛；它是显存上限/启动检查值，不会量化模型或改变 checkpoint。
- `--max_batch_size 1`：当前只有一个 G1 WebSocket session，使用串行推理并减少 KV cache，
  不为默认 8 路并发预留显存。

如果日志仍在 CUDA graph capture 或 warmup 阶段报告 OOM，可先用低显存 eager 模式验证链路：

```bash
cd /home/embodied/LightNav-0
conda activate lightnav-0
CUDA_VISIBLE_DEVICES=0 PORT=8051 VLN_VLLM_ENFORCE_EAGER=1 \
  lightnav-serve \
  --task vln \
  --model_path checkpoints/LightNav-0 \
  --backend vllm_local \
  --gpu_memory_utilization 0.70 \
  --max_batch_size 1
```

`VLN_VLLM_ENFORCE_EAGER=1` 会降低 CUDA graph 的额外显存和启动压力，但每步推理会更慢；
只有普通命令在后续阶段仍 OOM 时才需要使用。

## 9. 终端 E：启动 EMERGE Controller

```bash
cd /home/embodied/EMERGE-Humanoid
conda activate EmergePolicy
export PYTHONPATH=/home/embodied/EMERGE-Humanoid
python -m robot.controller \
  --driver lightnav_sonic_g1 \
  --workspace /home/embodied/.Emerge/workspace \
  --driver-config /home/embodied/EMERGE-Humanoid/integrations/lightnav_sonic_emerge/driver_procthor.json \
  --interval 0.25
```

参数含义：

- `--driver lightnav_sonic_g1`：加载本次实现的 external driver；
- `--workspace`：Controller 与 Main Agent 共用 `ACTION.md`、`ROBOT_STATE.md` 和观察文件；
- `--driver-config`：加载 LightNav、camera、yaw 和 SONIC 的 endpoint；
- `--interval 0.25`：无动作时每 0.25 秒检查一次动作队列。

Controller 启动时会把专用 profile 安装成 workspace 的 `EMBODIED.md`。它在真正执行
`navigate_vln` 时才绑定 `5556`、连接 LightNav，并在所有退出路径发送 locomotion `IDLE`。

必须先看到终端 E 输出以下关键信息，才能执行终端 F：

```text
Profile installed: profile_lightnav_sonic_g1.md -> .../EMBODIED.md
ROBOT_STATE.md updated.
Watching ACTION.md ...
```

可在另一终端做就绪检查：

```bash
rg -n 'Unitree G1|g1_sonic_001|lightnav_sonic_g1' \
  /home/embodied/.Emerge/workspace/EMBODIED.md \
  /home/embodied/.Emerge/workspace/ROBOT_STATE.md
```

若仍只看到 onboard 模板中的 `go2_edu_001` 或 `desktop_pet_001`，说明 Controller 尚未成功启动，
不要提前入队。

## 10. 终端 F：选择一种发起任务的方式

### 10.1 推荐先做：跳过 LLM，验证真实动作链路

```bash
cd /home/embodied/EMERGE-Humanoid
conda activate EmergePolicy
python -m integrations.lightnav_sonic_emerge.enqueue_navigation \
  --workspace /home/embodied/.Emerge/workspace \
  --instruction "Please move next to the table and stop" \
  --max-steps 120 \
  --control-hz 4
```

这不是 mock waypoint。它只跳过 Main Agent 的语言规划，后面的
`ACTION.md → Controller → 真实 LightNav 模型 → Adapter → SONIC → G1` 全部是真实链路。

- `--max-steps` 是最多推理/控制 tick 数，不是米；
- `120 / 4 Hz` 的名义控制窗口约 30 秒，实际墙钟时间还包含每次模型推理；
- 首版 Driver 将单次 action 限制为最多 120 ticks，长距离任务应观察后分段调用；
- 动作结束后可继续运行该命令发送下一条指令，不需要重启 A～E；
- 前一条仍为 `pending/running` 时，工具会拒绝叠加第二条动作。

查看结果：

```bash
sed -n '1,260p' /home/embodied/.Emerge/workspace/ACTION.md
sed -n '1,320p' /home/embodied/.Emerge/workspace/ROBOT_STATE.md
if test -f /home/embodied/.Emerge/workspace/artifacts/observations/observation.json; then
  python -m json.tool /home/embodied/.Emerge/workspace/artifacts/observations/observation.json
else
  echo "observation 尚未生成：检查 ACTION 状态和终端 E；Driver 必须先收到有效 camera + yaw"
fi
```

状态判断：

- `pending`：Controller 没有处理该动作，首先检查终端 E；
- `running`：动作已经进入 Driver，首个有效 camera + yaw 到达后才会生成 observation；
- `failed`：读取 action result 和终端 E 的具体错误；
- `completed`：底层动作结束，但仍需检查图像确认语义目标是否到达。

### 10.2 再做：由 EMERGE Main Agent 规划并调用 skill

交互模式：

```bash
cd /home/embodied/EMERGE-Humanoid
conda activate EmergePolicy
emerge \
  --workspace /home/embodied/.Emerge/workspace \
  --model openai-codex/gpt-5.1-codex
```

在界面中输入，例如：

```text
请控制机器人移动到桌子旁边并停下。先建立带有可观察完成条件的计划；导航动作结束后检查最新视觉证据，不能只根据 max_steps 用尽就宣布成功。
```

如使用的不是 OpenAI Codex，把 `--model` 替换成已配置 provider 对应的模型名称。

也可进行一次 headless 测试：

```bash
cd /home/embodied/EMERGE-Humanoid
conda activate EmergePolicy
python -m Emerge.cli.headless \
  "请控制机器人移动到桌子旁边并停下；用最新视觉证据验证是否真的到达。" \
  --workspace /home/embodied/.Emerge/workspace \
  --model openai-codex/gpt-5.1-codex \
  --timeout-s 300 \
  --cancel-timeout-s 15
```

## 11. 成功时应观察到什么

1. 终端 E 记录 `Action: 'navigate_vln'`，然后显示结构化执行结果。
2. 终端 D 持续收到同一 WebSocket session 的 RGB + instruction 推理请求。
3. 终端 A 创建 release file 后释放 Elastic Band，G1 接受 SONIC locomotion。
4. `ACTION.md` 从 `pending → running → completed/failed/cancelled`。
5. `ROBOT_STATE.md` 记录导航状态、tick、最后 waypoint 和 yaw；不会伪造当前接口没有提供的世界 X/Y。
6. `artifacts/observations/ego_view.png` 与 `observation.json` 持续更新。

注意：`completed` 仅表示一次底层导航动作结束；LightNav 的 `stop=true` 或 `max_steps` 用尽都不能单独证明“已到桌旁”。目标是否完成必须结合最新图像验证。

## 12. 停止与再次测试

- 同一轮仿真中：等上一条动作结束后，重复终端 F 的入队命令即可；不要删除 release file。
- 需要人工接管时：先关闭终端 E，确认 `5556` 已释放，再启动既有 teleop；teleop 退出后再启动终端 E。
- 整轮关闭建议顺序：先结束 F/Agent，再结束 E/Controller，然后 D/LightNav、B/SONIC，最后 A/MuJoCo。
- 下一轮重新启动 A 之前，再执行一次：

```bash
rm -f /tmp/lightnav_sonic_release
```

## 13. 当前验证边界

- 已完成：代码编译、41 项离线测试、插件注册及从非仓库目录加载 Driver。
- 已实测：真实 GPU LightNav server → Adapter → SONIC TensorRT → ProcTHOR MuJoCo G1；一次桌子任务在 50 ticks / 49 条 planner command 后收到 `LightNav stop=true`，机器人到达桌前。
- 尚未实测：Main Agent（Codex/API）自主规划、独立视觉成功验证及真机运行。因此仍建议先按 `10.1` 隔离验证真实导航 action，再测试 Main Agent。
