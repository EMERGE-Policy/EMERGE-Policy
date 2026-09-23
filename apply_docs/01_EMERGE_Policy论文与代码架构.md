# EMERGE-Policy 论文与代码架构梳理

## 1. 阅读范围与结论

本文基于以下材料：

- 论文：[EMERGE-Policy: A Robot Mind Emerges Beyond a Single Policy](https://arxiv.org/pdf/2608.29896)（arXiv v2，2026-09-09）。
- 当前 fork：`/home/embodied/EMERGE-Humanoid`，分支 `humanoid-dev`，本文分析时 commit 为 `1c6accd`。
- 导航原型最初位于 `/home/embodied/LightNav-0/lightnav_sonic_bridge/`；为保证分支可复现，实际接入版本已收敛到 `integrations/lightnav_sonic_emerge/bridge/`，不依赖 LightNav 仓库中的未跟踪脚本。
- SONIC：`/home/embodied/GR00T-WholeBodyControl`。

核心结论：EMERGE-Policy 不是一个新的低层机器人 policy，而是一个把大模型规划、感知、低层 policy、结果验证、恢复和记忆组织成闭环的 orchestration framework。LightNav-0 + SONIC 在该框架中最适合作为 **Operational Skill（执行型 Skill）**，通过现有的 `execute_robot_action → ACTION.md → Controller → Driver` 路径接入；不应该直接嵌入主 Agent 的推理循环，也不应该把 LightNav 当成 Sub-Agent。

## 2. 论文中的系统设计

### 2.1 Main Agent

Main Agent 保存紧凑的任务级状态，并负责：

1. 把用户任务拆成分层 subgoal。
2. 为每个 subgoal 明确三元组：
   - `Target`：目标实体；
   - `Change`：希望发生的状态变化；
   - `Criterion`：可观察、可验证的完成条件。
3. 选择要调用的 Sub-Agent 或 Skill。
4. 根据执行和验证结果推进计划、局部恢复或全局重规划。

论文将主 Agent 的动作抽象为：委派 Sub-Agent、调用 Skill、更新状态、恢复、重规划或终止。

### 2.2 Hierarchical Task Planner 与 Branch Stack

正常任务步骤保存在 Main Line 中。某一步失败时，Main Agent 产生文本化 failure diagnosis，并把局部恢复步骤压入 Branch Stack：

```text
Main Line: S1 → S2 → S3
                 ↑
                 └─ Branch Stack: recovery-1 → recovery-2
```

优先执行栈顶 recovery；恢复完成后出栈并回到被中断的主线。只有局部恢复反复失败时，才修改高层计划。这个机制很适合未来的主动导航，例如：

```text
主线：走到沙发前
失败诊断：目标不在当前视野，连续前进无有效进展
恢复分支：原地扫描 → 重新观察 → 从新朝向继续导航
```

### 2.3 Sub-Agent Suite

论文把证据密集型工作放入隔离上下文，主 Agent 只接收结构化结论：

- Perception Sub-Agent：目标识别、定位、空间关系和 affordance。
- Verification Sub-Agent：检查前置条件及 subgoal 完成条件。
- Monitor Sub-Agent：执行中持续观察异常并触发提前停止。
- Memory Integration：上下文过长时压缩并整合长期记忆。

Sub-Agent 的价值不是代替低层控制，而是隔离大量图片、mask、点云和中间推理，避免污染 Main Agent 的上下文。

### 2.4 Unified Skill Library

论文按“在闭环中的功能”而不是按模型结构划分 Skill：

| 类型 | 功能 | 典型实现 | 本项目映射 |
|---|---|---|---|
| Operational Skill | 产生并执行动作 | motion primitive、VLA、WAM policy | LightNav-0 + Adapter + SONIC |
| Imagination Skill | 执行前预测候选结果 | world model、视频预测 | 暂不接入 |
| Evaluation Skill | 评估可行性、进度与完成条件 | VLM、几何检查、环境状态 | 当前 Task Verification；后续导航 verifier |

论文中的统一接口可概括为：

```text
K = <能力摘要, 参数模式, 调用前置条件, 预期后置条件>
```

LightNav 导航 Skill 因此不能只有一个“启动脚本”，还应明确输入、完成条件、取消行为、失败类型和运行状态回写。

### 2.5 闭环生命周期

论文给出的完整生命周期是：

```text
Planning
  ↓
Perceptual grounding
  ↓
Physical dispatch
  ↓
Outcome verification
  ├─ success → 推进 PLAN
  └─ failure → Branch Stack recovery / replan
```

对导航任务可映射为：

```text
“走到桌子旁”
  ↓
Main Agent：建立导航 subgoal 和完成条件
  ↓
navigate_vln Operational Skill
  ↓
LightNav waypoint → SONIC locomotion
  ↓
第一视角观察 / LightNav stop / 超时 / 异常
  ↓
完成、恢复扫描或重规划
```

## 3. 当前仓库的代码地图

```text
EMERGE-Humanoid/
├── Emerge/
│   ├── agent/
│   │   ├── loop.py                 # Main Agent 的 LLM/tool 循环
│   │   ├── context.py              # 组装 system prompt 和 workspace 状态
│   │   ├── memory.py               # token 阈值与 MEMORY.md 整合
│   │   ├── plan_state.py           # 读取当前主线/Branch Stack subgoal
│   │   ├── visual_monitor.py       # 后台视觉监控与动作中断
│   │   ├── skills.py               # 内置和 workspace SKILL.md 加载
│   │   └── tools/
│   │       ├── embodied.py         # execute_robot_action，写 ACTION.md 并等结果
│   │       ├── update_plan.py      # 维护 PLAN.md
│   │       ├── delegate.py         # 调用注册的 Sub-Agent
│   │       ├── scene_graph.py      # 查询 ROBOT_STATE.md
│   │       └── filesystem/shell/message
│   ├── base/
│   │   ├── tool.py                 # Tool 抽象接口
│   │   └── registry.py             # ToolRegistry
│   ├── subagents/
│   │   ├── base.py                 # 隔离的 Sub-Agent LLM/tool loop
│   │   ├── registry.py             # SubagentRegistry
│   │   ├── object_location/        # VGGT + SAM3 多视角目标定位
│   │   └── task_verification/      # 多视角可见状态验证
│   ├── skills/                     # 主 Agent 内置使用说明
│   ├── providers/                  # LiteLLM、Codex OAuth 等 provider
│   ├── runtime/                    # headless run、租约、事件和产物
│   ├── session/                    # session JSONL/history
│   ├── config/                     # 配置 schema 和加载
│   └── templates/                  # workspace 初始文件模板
├── robot/
│   ├── controller.py               # ACTION.md watchdog 与 driver 调度
│   ├── drivers/
│   │   ├── base_driver.py          # BaseDriver 接口
│   │   └── mujoco_driver.py        # 内置 LIBERO driver
│   ├── plugins.py                  # external driver manifest/registry
│   ├── profiles/                   # EMBODIED.md robot profile
│   ├── mujoco_simulation/          # LIBERO 相机、状态、仿真封装
│   └── vla/                        # OpenPI client/executor
├── external_model_server/          # OpenPI、VGGT、SAM3 server
├── scripts/                        # benchmark 与 model-server 启动脚本
├── dev/                            # driver/eval JSON 示例
└── third_party/                    # git submodules
```

## 4. 当前代码的实际运行链路

### 4.1 Main Agent 初始化

`Emerge/runtime/service.py::build_agent()` 创建 `AgentLoop`，注入 provider、workspace、模型、tool 限制、Sub-Agent 配置和 Visual Monitor 配置。

`AgentLoop._register_default_tools()` 注册：

- 文件读写；
- `update_plan`；
- shell；
- message；
- `delegate_subagent`；
- `execute_robot_action`；
- `query_scene_graph`。

当前注册的 specialist 是 `object_location` 和 `task_verification`。

### 4.2 Context 与 Skill 加载

`ContextBuilder` 每轮将以下内容组装给 Main Agent：

```text
身份与运行环境
+ workspace/AGENTS.md
+ workspace/EMBODIED.md（存在时）
+ workspace/ROBOT_STATE.md（存在时）
+ workspace/PLAN.md（存在时）
+ memory/MEMORY.md
+ always skills 全文
+ 其他 skills 的摘要和路径
+ session history
+ 当前用户消息
```

主 Agent Skill 有两层来源：

1. `Emerge/skills/<name>/SKILL.md`：内置 skill；
2. `<workspace>/skills/<name>/SKILL.md`：workspace skill，重名时优先。

这里的 `SKILL.md` 是“如何选择和调用能力”的说明，并不自动创建新的 Controller action handler。

### 4.3 Action Queue 与 Controller

`execute_robot_action` 不直接控制机器人，而是：

1. 检查 workspace 中是否存在 `EMBODIED.md`。
2. 将 `action_type + parameters` 原子写入 `ACTION.md`。
3. 等待 action 从 `pending → running → completed/failed/cancelled`。
4. 必要时写入 `cancel_requested`。

`robot/controller.py` 轮询同一份 `ACTION.md`：

```text
pending action
  ↓ claim
running
  ↓ driver.execute_action(action_type, params, cancel_check)
terminal result
  ↓
更新 ROBOT_STATE.md 和 ACTION.md
```

因此 Controller/Driver 是接入任何实际机器人能力的稳定边界。

### 4.4 BaseDriver 与 external plugin

所有 driver 必须实现：

```python
get_profile_path() -> Path
load_scene(scene: dict) -> None
execute_action(action_type, params, cancel_check=None) -> str
get_runtime_state() -> dict
close() -> None
```

`robot/plugins.py` 支持通过 `Emerge_plugin.toml` 注册外部 driver。Controller 根据 manifest 动态加入 `sys.path` 并加载 `BaseDriver` 子类。这意味着 LightNav+SONIC 可以放在新的 `integrations/` 目录中，无需修改内置 `DRIVER_REGISTRY`。

### 4.5 视觉观察、验证和提前停止

Task Verification 读取：

```text
workspace/artifacts/observations/observation.json
```

其中列出当前图片路径与 reference view。它只需要当前图像；Object Location 额外要求每个视角的相机内参与 `T_world_camera`。

Visual Monitor 是当前代码中真正的后台执行模块：它轮询 `PLAN.md`、`ACTION.md` 和 observation manifest，发现当前 subgoal 已完成时，向 `EmbodiedActionTool` 发出中断，后者在 `ACTION.md` 中请求 Controller 取消当前动作。

### 4.6 状态与记忆文件

| 文件 | 当前代码用途 |
|---|---|
| `AGENTS.md` | 全局 orchestration 指令 |
| `EMBODIED.md` | 当前 driver 支持的动作与约束 |
| `ROBOT_STATE.md` | Controller 回写 robot/scene/nav runtime state |
| `PLAN.md` | Main Line、Pointer、Retries 与 Branch Stack |
| `ACTION.md` | Agent 与 Controller 之间的原子动作队列 |
| `memory/MEMORY.md` | token-aware 长期记忆整合结果 |
| `sessions/*.jsonl` | 会话历史 |
| `artifacts/observations/observation.json` | 当前相机快照清单 |

## 5. 论文描述与当前实现的差异

这些差异决定了接入时不能只按论文图示假设接口：

1. **Sub-Agent 并非全部后台运行。** 当前 `delegate_subagent` 是 async coroutine，但 Main Agent 会 `await` 其结果；Visual Monitor 才是独立后台线程。
2. **当前没有独立 `HISTORY.md`。** 会话保存在 session 文件中，长期事实写入 `memory/MEMORY.md`。
3. **Branch Stack 没有独立文件。** 它是 `PLAN.md` 数据结构的一部分。
4. **Skill interface 尚非统一 Python backend registry。** 论文中的 Operational/Imagination/Evaluation 是功能抽象；当前仓库中，动作 backend 由 Controller driver 实现，`SKILL.md` 主要提供调用策略。
5. **`execute_robot_action` 的说明目前只列 manipulation action。** 其 JSON schema 中 `action_type` 仍是自由字符串，所以 workspace 的 `EMBODIED.md` 和 navigation `SKILL.md` 可以先指导模型调用 `navigate_vln`，无需第一阶段修改核心 tool；后续可再做 capability-driven 动态 schema。
6. **内置 object localization 面向 LIBERO 多标定相机。** 当前 G1 只有一个已发布的 `head_camera`，且尚未提供可靠相机外参与多视角重建，因此第一阶段不应把 VGGT/SAM3 object-location 作为导航前置条件。

## 6. 对导航接入的架构判断

### 6.1 正确角色

- Main Agent：把长任务拆成“导航到目标附近”“确认目标可见/已到达”等 subgoal。
- Navigation `SKILL.md`：告诉 Main Agent 何时调用、参数如何写、何时停止、失败后如何恢复。
- SONIC external driver：执行 `navigate_vln`，维护取消、安全停止和 runtime state。
- LightNav-0：视觉语言导航 policy，输出局部 waypoint。
- Adapter：坐标、速度、停止和异常安全转换。
- SONIC：Kinematic Planner + WBC，产生人形 locomotion。
- Task Verification / 后续导航 verifier：检查是否真正到达，而不是把 action 返回当成任务成功。

### 6.2 不应采用的接法

- 不把 LightNav waypoint 直接交给 Main Agent 逐个解释；这会增加延迟并污染上下文。
- 不让 Main Agent 用 shell 启动一个无限 runner；这绕开 ACTION 状态、取消和 Visual Monitor。
- 不让独立 bridge runner 与 EMERGE driver 同时 bind `tcp://*:5556`。
- 不把 LightNav 当成 Sub-Agent；它是产生控制意图的 Operational Skill backend。
- 不在第一阶段修改 SONIC planner/WBC 或让其直接理解完整 waypoint sequence。

## 7. 当前架构的下一步扩展方向

第一阶段只接 `navigate_vln`，复用已验证的 first-waypoint adapter。第二阶段增加显式 `scan_environment`/`turn_in_place` 作为恢复 primitive，并将第一视角快照写入 observation manifest，使 Visual Monitor 能提前终止。第三阶段再加入导航专用 Evaluation Skill、语义场景记忆和跨 skill 的“导航到操作位姿 → GR00T manipulation”组合。
