# LightNav-0 + SONIC 导航 Skill：实现文件与改动说明

## 1. 改动范围

本次只在 `/home/embodied/EMERGE-Humanoid` fork 内新增独立 integration、测试和文档。为了让其他开发者仅克隆本分支即可获得完整接入代码，原型 bridge 已作为本 integration 的自包含模块提交。

没有修改：

- `/home/embodied/LightNav-0` 的官方已有源码、checkpoint 或模型配置；
- `/home/embodied/GR00T-WholeBodyControl` 的 SONIC planner、WBC、deployment 或场景；
- EMERGE 的 `Emerge/`、`robot/` 等核心已有文件。

## 2. Integration 文件

目录：`integrations/lightnav_sonic_emerge/`

| 文件 | 作用 |
|---|---|
| `Emerge_plugin.toml` | 声明 external driver `lightnav_sonic_g1`，使用插件目录内入口，支持从任意工作目录加载。 |
| `driver.py` | 实现 `SonicLightNavDriver(BaseDriver)`；校验 `navigate_vln` 参数、启动有界 backend、回写 runtime state、区分底层动作结束与语义目标成功。 |
| `navigation_backend.py` | 执行低层闭环；订阅 camera/yaw、连接 LightNav、调用 integration 内 adapter、发布 SONIC packet、处理取消/stop/陈旧输入/异常，并在 `finally` 中发送 IDLE。 |
| `observation_writer.py` | 原子写入最新 `ego_view.png` 和 EMERGE Visual Monitor/Task Verification 可读的 `observation.json`。 |
| `enqueue_navigation.py` | 不调用 LLM 的确定性测试入口；把 `navigate_vln` 安全写入标准 `ACTION.md` 队列，并拒绝并发动作。 |
| `profile_lightnav_sonic_g1.md` | Controller 安装到 workspace 的 embodiment profile；声明传感器、action、验证规则和安全限制。 |
| `driver_procthor.json` | ProcTHOR 测试的 endpoint、频率、步数和 Elastic Band 配置；不包含本机绝对 workspace 路径。 |
| `workspace_skill/SKILL.md` | 告诉 Main Agent 如何计划、调用和验证导航 skill，禁止把 action completion 直接当作目标成功。 |
| `README.md` | 面向开源协作者的架构、环境、复现、运行、测试、限制及上游归属说明。 |
| `__init__.py` | Python package 标识和公开 Driver。 |

`bridge/` 内包含 `adapter.py`、`lightnav_ws.py`、`sonic_zmq.py`、`runner.py`、`room_sim.py`、场景合成与 teleop 等独立实现。它只读取官方仓库的运行时资产，通过 WebSocket/ZMQ 对接，不修改两边 policy。

仓库根目录还新增 `integrations/__init__.py`，用于常规包导入与单元测试。

## 3. 测试文件

目录：`integrations/lightnav_sonic_emerge/checks/`。仓库现有 `.gitignore` 会忽略名为
`tests` 的目录，因此这里使用 `checks`，无需修改仓库级忽略规则。

| 文件 | 覆盖内容 |
|---|---|
| `test_driver.py` | action/参数校验、执行结果、runtime state、不得伪造 X/Y。 |
| `test_navigation_backend.py` | LightNav stop、步数边界、取消、陈旧输入、Elastic Band release、最终 IDLE 和资源关闭。 |
| `test_observation_writer.py` | JPEG→PNG、manifest、revision、禁用与空输入。 |
| `test_plugin.py` | manifest 注册和 external Driver 实例化。 |
| `test_enqueue_navigation.py` | 标准 action 入队与并发动作拒绝。 |
| `bridge/test_adapter.py` | 坐标旋转、符号、速度裁剪、近点停止、纯转向及无效输入 fallback。 |
| `bridge/test_lightnav_ws.py` | WebSocket reply 的 waypoint/stop 严格解析。 |
| `bridge/test_runner.py` | camera payload、G1 四元数 yaw 和 runner 安全条件。 |
| `bridge/test_sonic_zmq.py` | SONIC command/planner 二进制 packet 字段。 |
| `bridge/test_teleop.py` | 键盘运动状态到 SONIC command 的转换。 |
| `__init__.py` | 测试包标识。 |

当前共有 41 项测试，均已通过。

## 4. 文档文件

目录：`apply_docs/`

- `01_EMERGE_Policy论文与代码架构.md`：论文及仓库架构。
- `02_环境配置.md`：当前目标所需的最小环境配置。
- `03_LightNav0_SONIC导航Skill接入方案.md`：接入设计。
- `04_导航Skill执行与测试终端指令.md`：从初始化到六终端联合测试的完整命令。
- `05_导航Skill实现文件与改动说明.md`：本文件，记录实际落地范围。

## 5. 实际调用关系

```text
execute_robot_action / enqueue_navigation
  → ACTION.md
  → robot.controller（已有，无修改）
  → SonicLightNavDriver.execute_action
  → NavigationBackend.run
  ├─ LightNavWsClient（integration/bridge）
  ├─ SonicCameraSubscriber / SonicYawSubscriber（integration/bridge）
  ├─ LightNav2SonicAdapter（integration/bridge）
  ├─ SonicZmqPublisher（integration/bridge）
  └─ EgoObservationWriter（新增）
  → SONIC 标准 planner packet
```

## 6. 已验证和未验证

已验证：

- Python 编译；
- 41 项离线单元与协议测试；
- 插件 manifest 注册；
- 从 `/tmp` 等非仓库目录加载 `driver.SonicLightNavDriver`；
- action queue、取消、安全停止、结果和观察文件的数据契约。

尚未在本次实现过程中宣称完成：

- Main Agent 对实际目标到达的稳定视觉判断；
- 主动环视、显式原地转向 recovery primitive；
- 可信世界 X/Y 定位或 SLAM。

另外已完成真实 GPU LightNav-0 → SONIC TensorRT → ProcTHOR G1 的实时联合测试：桌子指令在 50 ticks、49 条 planner command 后由 LightNav 返回 stop，G1 到达桌前。该结果验证的是确定性入队后的真实模型链路，尚未包含 Main Agent。运行和复现请使用 `04_导航Skill执行与测试终端指令.md`。
