# Emerge 包说明

`Emerge` 是项目的主 Agent 包，负责组织 LLM 上下文、skills、tools、会话状态、具身动作，以及已经注册的专业子 Agent。机器人和仿真环境由仓库根目录下的 `robot/` 运行，二者通过 workspace 文件交互。

## 1. 核心结构

```text
用户任务
   ↓
AgentLoop
   ├── ContextBuilder：AGENTS.md、具身状态、memory、skills
   ├── ToolRegistry：文件、Shell、计划、消息、具身动作、场景查询
   ├── SessionManager：会话历史
   └── delegate_subagent
          ↓
      SubagentRegistry
          ├── ObjectLocationSubagent
          ├── 私有 context
          ├── object-localization skill
          ├── observe_scene
          ├── segment_candidates → VGGT + SAM3 + 完整视角 overlays
          └── locate_candidates → 复用已确认候选的缓存 geometry/masks
          └── TaskVerificationSubagent
              ├── 私有 context
              ├── object-state-verification skill
              ├── observe_scene
              └── submit_task_verification → 提交可见状态与证据
```

主 Agent 不直接操作子 Agent 的内部工具。动作前通过 `object_location` 获取精确世界坐标；
动作后通过 `task_verification` 验证目标物体状态是否真正达成。

## 2. 模块地图

```text
Emerge/
├── __main__.py                    # python -m Emerge
├── agent/
│   ├── loop.py                    # 主 Agent 推理与工具循环
│   ├── context.py                 # 主 Agent system prompt
│   ├── memory.py                  # 长程记忆与上下文压缩
│   ├── skills.py                  # 内置和 workspace skills 加载
│   └── tools/
│       ├── delegate.py            # 调用注册型专业子 Agent
│       ├── embodied.py            # 写入机器人动作
│       ├── filesystem.py          # read / write / edit / list
│       ├── message.py             # 当前任务进度消息
│       ├── scene_graph.py         # 查询 ROBOT_STATE.md
│       ├── shell.py               # workspace Shell 工具
│       └── update_plan.py         # PLAN.md 状态更新
│
├── base/
│   ├── tool.py                    # Tool 抽象接口
│   └── registry.py                # ToolRegistry
│
├── bus/                           # CLI 与 AgentLoop 使用的消息队列
├── cli/commands.py                # onboard / agent / status / provider
├── config/                        # 配置 schema、加载与运行路径
├── providers/                     # LiteLLM、Azure、Codex、兼容端点
├── session/                       # workspace/sessions/*.jsonl
├── skills/                        # 主 Agent 内置 skills
│   ├── object-location/           # 何时委派精确定位
│   └── task-verification/         # 何时委派动作结果验证
├── templates/                     # workspace 初始模板
│
└── subagents/
    ├── base.py                    # BaseSubagent 公共执行循环
    ├── content.py                 # 文本、图片等多模态输入
    ├── context.py                 # 每次调用的独立上下文
    ├── models.py                  # 任务、描述符、任意结构结果
    ├── registry.py                # 完整子 Agent 实例注册表
    ├── skills.py                  # 子 Agent 私有 SkillRegistry
    └── object_location/
        ├── main.py                # 独立对话入口
        ├── register.py            # 组装并注册实例内部能力
        ├── agent.py / context.py
        ├── skills/object-localization/SKILL.md
        └── tools/                 # 观测、候选验证和定位工具
    └── task_verification/
        ├── main.py                # 独立对话入口
        ├── register.py            # 组装验证实例
        ├── agent.py / context.py
        ├── skills/object-state-verification/SKILL.md
        └── tools/                 # 多视角观测和结构化结果提交
```

## 3. 运行配置

默认配置文件：

```text
~/.Emerge/config.json
```

默认 workspace：

```text
~/.Emerge/workspace
```

两个视觉子 Agent 的配置位于 `subagents.objectLocation` 和
`subagents.taskVerification`：

```json
{
  "subagents": {
    "objectLocation": {
      "model": "openai/gpt-5.5",
      "vggtUrl": "ws://localhost:8001",
      "sam3Url": "ws://localhost:8002",
      "timeout": 120.0,
      "maxIterations": 8,
      "viewCenterToleranceM": 0.08,
      "rayConsensusToleranceM": 0.02
    },
    "taskVerification": {
      "model": "openai/gpt-5.5",
      "maxIterations": 4
    }
  }
}
```

运行时采用 camelCase 配置键；Python 中对应 `config.subagents.object_location` 和
`config.subagents.task_verification`。
`viewCenterToleranceM` 用于 VGGT 深度中心的后备一致性检查；精度更高的标定射线共识使用
独立的 `rayConsensusToleranceM`。

## 4. 常用命令

从仓库根目录安装：

```bash
pip install -e .
```

项目要求 Python 3.10 或更高版本。首次创建或刷新配置：

```bash
python -m Emerge onboard
# 等价：emerge onboard
```

启动主 Agent 交互终端：

```bash
python -m Emerge agent
# 等价：emerge agent
```

单次执行任务：

```bash
python -m Emerge agent -m "Find the salad dressing"
```

指定配置、workspace 和 session：

```bash
python -m Emerge agent \
  --config ~/.Emerge/config.json \
  --workspace ~/.Emerge/workspace \
  --session cli:direct
```

检查当前配置状态：

```bash
python -m Emerge status
```

OAuth provider 登录：

```bash
python -m Emerge provider login openai-codex
python -m Emerge provider login github-copilot
```

当前不存在 `chat` 和 `gateway` 子命令；交互入口统一使用 `agent`。

## 5. 外部模型服务

OpenPI、VGGT 和 SAM3 都位于仓库根目录的 `external_model_server/`，统一启动命令为：

```bash
bash scripts/model_server/start_external_model_servers.sh
```

默认分配如下：

| 服务 | 端口 | conda 环境 |
|---|---:|---|
| OpenPI | 8000 | `pi05_server` |
| VGGT | 8001 | `EmergePolicy` |
| SAM3 | 8002 | `EmergePolicy` |

GPU 和 OpenPI batch 参数可以通过环境变量覆盖：

```bash
OPENPI_GPU=3,4,5 \
OPENPI_MAX_BATCH_SIZE=1 \
VGGT_GPU=1 \
SAM3_GPU=2 \
bash scripts/model_server/start_external_model_servers.sh
```

## 6. 单独运行 Object Location Subagent

交互模式：

```bash
python -m Emerge.subagents.object_location.main
```

单次任务：

```bash
python -m Emerge.subagents.object_location.main \
  --task "Find the salad dressing"
```

还可以使用 `--config`、`--workspace` 和 `--model` 覆盖运行参数。该实例要求 workspace 中已经存在：

```text
artifacts/observations/observation.json
```

清单中的每个启用视角都应提供图片路径、`intrinsics` 和 `T_world_camera`。Controller 逐相机覆盖保存图片并覆盖更新该清单；定位时由整个清单参与 VGGT 重建，再在合并目标点云时选择一致视角。

当前定位数据流统一使用相机原始 `512×512` RGB：多模态模型查看原图，SAM3 在原图上分割，VGGT 输出同尺寸 depth 和 point map，mask overlay 也保持 512。VGGT 内部仅为满足 14 像素 patch 要求把右侧和下侧 padding 到 518，推理后立即裁回 512，不改变内参和像素坐标。

子 Agent 只读取当前完整视角，不读取物体参考图或 `ROBOT_STATE.md`。多模态模型根据用户
给出的语义目标和现场可见证据比较所有合理物理候选，再为各候选生成纯外观 SAM3 prompt。
`segment_candidates` 一次分割全部候选，把带 `candidate_id` 的 mask overlay 作为多模态
结果返回。子 Agent 回看每个视角，提交同一物体真正匹配的 `verified_views`，再调用
`locate_candidates`。

最终定位使用已知相机标定下的 bbox 射线共识剔除跳到其他物体的 mask，以三角化中心校正
各视角 VGGT 局部点云，再做姿态估计。最后一步复用本轮缓存的 VGGT geometry 和 SAM3
masks，不会再次请求两个模型 server。

## 7. 单独运行 Task Verification Subagent

交互模式：

```bash
python -m Emerge.subagents.task_verification.main
```

单次验证：

```bash
python -m Emerge.subagents.task_verification.main \
  --task "Verify whether the apple is inside the basket and released"
```

该实例读取同一份 `artifacts/observations/observation.json` 和当前完整视角，将目标结果拆成
可见条件，再通过 `submit_task_verification` 提交证据。它不调用 VGGT、SAM3 或坐标定位，
整体结果由代码根据所有必需条件计算为 `achieved`、`not_achieved` 或 `uncertain`。

## 8. 主 Agent 调用子 Agent

主 Agent 有两个对应的常驻 skills：`object-location` 指导动作前的精确定位，
`task-verification` 指导动作后的结果检查。两者都通过 `delegate_subagent` 调用完整子 Agent，
但分别使用 `object_location` 和 `task_verification` 两个注册名。

主 Agent 注册的工具名为 `delegate_subagent`。模型发起的内部参数形如：

```json
{
  "agent_name": "object_location",
  "task": "Locate the salad dressing precisely and describe nearby obstacles."
}
```

该调用是异步 coroutine，但当前主 Agent 回合会 `await` 定位结果。它不会阻塞整个 asyncio 事件循环，也不是启动后立即返回的后台任务。

返回内容包含：

```json
{
  "status": "success",
  "agent_name": "object_location",
  "summary": "Localized: salad_dressing.",
  "output": {
    "objects": [
      {
        "name": "salad_dressing",
        "found": true,
        "frame": "world",
        "position_m": [0.0997, -0.1923, 0.0503],
        "size_m": [0.0469, 0.0367, 0.1176],
        "rpy_rad": [0.0, 0.0, -1.10]
      }
    ],
    "scene_context": "The target is near several containers and a woven basket."
  },
  "error": null
}
```

动作后的验证调用示例：

```json
{
  "agent_name": "task_verification",
  "task": "Verify whether the apple is inside the basket and has been released by the gripper."
}
```

对应的精简结果形如：

```json
{
  "status": "success",
  "agent_name": "task_verification",
  "summary": "Verification outcome: achieved.",
  "output": {
    "outcome": "achieved",
    "predicates": [
      {
        "name": "apple_inside_basket",
        "value": true,
        "evidence": "The apple is visibly below the basket rim."
      },
      {
        "name": "apple_released",
        "value": true,
        "evidence": "The open gripper is separated from the apple."
      }
    ],
    "scene_context": "The basket remains upright."
  },
  "error": null
}
```

主 Agent 只接收上述精简字段。定位过程的 SAM prompt、候选 ID、视角选择、射线残差、点云
数量和几何对齐信息保留在 `object_location` 内部。两个子 Agent 都不直接控制机器人，主 Agent
根据定位或验证结果继续规划并调用具身动作工具。

同一动作阶段需要定位多个相关物体时，应在一次 `task` 中一起委派，使它们共享同一组观测和
重建结果。返回 `found: false` 时，主 Agent 必须把位置视为未知，不能自行猜测世界坐标。
定位和动作后验证应分成两次委派，因为两者对应不同的物理场景时刻。验证结果为
`not_achieved` 时继续修正，为 `uncertain` 时补充观察；两者都不能作为任务成功。

## 9. Workspace 契约

| 文件 | 用途 |
|---|---|
| `AGENTS.md` | 主 Agent 指令 |
| `EMBODIED.md` | 当前具身系统能力说明 |
| `ROBOT_STATE.md` | Controller 回写的机器人与场景运行状态 |
| `PLAN.md` | 主 Agent 当前任务状态 |
| `ACTION.md` | Controller 消费的动作队列 |
| `memory/MEMORY.md` | 长程记忆 |
| `artifacts/observations/observation.json` | 多相机图片与标定清单 |
| `sessions/*.jsonl` | 会话历史 |

## 10. 扩展方式

- 新增主 Agent tool：在 `agent/tools/` 实现 `Tool`，并在 `AgentLoop._register_default_tools()` 注册。
- 新增主 Agent skill：创建 `skills/<name>/SKILL.md`。
- 新增专业子 Agent：在 `subagents/<name>/` 放置自己的 agent、context、skills、tools 和 `register.py`，组装完成后把实例注册到主 Agent 的 `SubagentRegistry`。
- 新增 provider：扩展 `providers/registry.py`、provider 实现和 `config/schema.py`。
- 新增 workspace 契约：添加模板，并按需要加入 `ContextBuilder` 的加载列表。

代码行为与文档不一致时，以当前代码为准。
