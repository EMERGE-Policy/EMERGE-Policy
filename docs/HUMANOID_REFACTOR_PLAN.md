# EMERGE-Humanoid 当前重构方案

## 1. 当前唯一目标

先保持已有功能不变，把 BEHAVIOR-1K 接入整理成与仓库原生 LIBERO
评测相同的代码组织，然后重新验证一个 `turning_on_radio` instance 能完整运行。

当前阶段不建立通用 `BenchmarkAdapter`、`PolicyBackend`、experiment factory
或额外包装接口。只有在第二个 benchmark、第二个机器人或第二种策略真正接入时，
再从两条真实链路中提取共用接口。

仓库的长期定位仍然是面向人形机器人的 EMERGE 操作框架。BEHAVIOR-1K
只是当前第一个人形评测环境，π0.5 只是当前使用的低层策略，
`turning_on_radio` 只是第一个打通的任务。

## 2. 以 LIBERO 原生链路为参照

LIBERO 当前由以下三层构成：

1. `scripts/Libero_eval/` 负责评测参数、episode workspace、Agent 进程和结果；
2. `robot/drivers/mujoco_driver.py` 是 Controller 直接加载的原生 driver；
3. `dev/libero_agent_eval.json` 保存 driver 的基础配置。

BEHAVIOR 使用相同边界：

1. `scripts/Behavior_eval/` 负责单 episode 的启动、进程监督和结果；
2. `robot/drivers/behavior_r1pro_driver.py` 是 Controller 直接加载的原生 driver；
3. `dev/behavior_agent_eval.json` 保存 driver 的基础配置；
4. `robot/behavior/` 保存 BEHAVIOR websocket 协议、状态校验和相机帧发布；
5. `robot/profiles/behavior_r1pro.md` 保存 Controller 安装到 workspace 的机器人 profile。

BEHAVIOR driver 在 `robot/drivers/__init__.py` 中以 `behavior_r1pro` 静态注册，
不再依赖 EMERGE plugin 注册流程。

## 3. `integrations/` 的处理

`integrations/` 目录暂时保留。

- `integrations/behavior_b1k_pi05/` 完全删除；BEHAVIOR 已成为仓库内的原生评测链路。
- `integrations/lightnav_sonic_emerge/` 暂时原样保留，后续单独决定它是否迁入原生 robot/skill 结构。
- 不为了目录整齐移动 LightNav，也不修改其运行逻辑。

## 4. 为什么 BEHAVIOR 仍然需要多个进程

LIBERO 的 MuJoCo driver 可以在 Controller 进程中直接创建环境。BEHAVIOR 的
Isaac Sim 和官方 evaluator 必须运行在专用 Python 环境中，所以不能把所有逻辑
塞进 Controller 进程。

一次 BEHAVIOR episode 包含：

| 进程 | Python 环境 | 职责 |
|---|---|---|
| runner、Agent、Controller | `phyagentos` | LLM 决策、ACTION 队列、EMERGE driver |
| OpenPI server | `/data/yxy/openpi/.venv` | π0.5 推理 |
| BEHAVIOR evaluator | `/data/yxy/envs/behavior` | Isaac Sim、reset、step、成功条件和评分 |

`behavior_r1pro` driver 是 Controller 中的原生 driver，但其内部通过 websocket
控制另一个环境中的 evaluator。跨进程只是运行时约束，不再等于外部 plugin。

## 5. LLM 思考时的仿真语义

必须保留现有动作门控：

1. Agent 思考期间没有新的 `vla_execute` action；
2. evaluator 等待 driver 授予动作预算；
3. evaluator 不调用 `env.step`；
4. Isaac Sim 的仿真步数、物理时间和机器人状态都不推进。

因此这里的“暂停”是仿真时间冻结，不是让仿真持续运行并反复发送零动作。
LLM 的墙上时间耗时不会消耗 `--max-steps`。

## 6. 当前真实执行链路

```text
scripts/Behavior_eval/eval_behavior_agent.py
        │
        ├── 启动 OpenPI π0.5 server
        ├── 启动 robot.controller --driver behavior_r1pro
        ├── 启动官方 BEHAVIOR evaluator
        └── 启动 Emerge Agent
                    │
                    ▼
             execute_robot_action
                    │
                    ▼
                 ACTION.md
                    │
                    ▼
          BehaviorR1ProDriver
                    │
                    ▼
       websocket action-budget gate
                    │
                    ▼
       BEHAVIOR evaluator / Isaac Sim
```

runner 从 `dev/behavior_agent_eval.json` 读取基础配置，再为本轮 episode
生成输出目录内的 `driver_config.json`。episode workspace、端口、状态文件、
动作预算等运行时值只写入这份 episode 配置。

## 7. 当前代码边界

### `Emerge/`

继续只负责 Agent、LLM provider、memory、tool loop 和 session。不加入
BEHAVIOR、R1Pro、π0.5 或具体任务逻辑。

### `robot/controller.py`

继续负责消费 `ACTION.md`、调用 driver 和更新 `ROBOT_STATE.md`。它不知道
BEHAVIOR websocket 协议。

### `robot/drivers/behavior_r1pro_driver.py`

负责把 EMERGE 的 `vla_execute` action 映射到受控的 BEHAVIOR 执行预算，并把
官方 evaluator 状态发布到 `ROBOT_STATE.md`。它不启动 Agent，也不决定评分。

### `robot/behavior/`

负责当前 BEHAVIOR evaluator 与 OpenPI server 之间的协议兼容、动作门控、
状态文件校验和 observation 快照。这里允许出现 BEHAVIOR 语义，因为它就是
benchmark 传输层。

### `scripts/Behavior_eval/`

负责启动顺序、不同 Python 环境、端口、超时、workspace、日志和最终结果。
它是评测入口，不是新的框架层。

## 8. 本轮重构的验收条件

本轮完成必须同时满足：

- `integrations/behavior_b1k_pi05/` 不存在；
- `integrations/lightnav_sonic_emerge/` 未被改动；
- 仓库中没有代码继续导入旧 BEHAVIOR integration；
- Controller 能通过原生注册名 `behavior_r1pro` 加载 driver；
- runner 能读取 `dev/behavior_agent_eval.json` 并生成 episode config；
- 旧的 BEHAVIOR driver、gateway、observation、action queue 和 runner 离线测试通过；
- `scripts/Behavior_eval/eval_behavior_agent.py --help` 可直接运行；
- 不调用外部 LLM、不启动 Isaac Sim 的验证全部通过。

完成离线验收后，再由用户决定何时执行一个真实 instance。真实验证命令与
产物说明写在 `scripts/Behavior_eval/README.md`。

## 9. 后续顺序

1. 真实运行 `turning_on_radio` 的一个 instance，确认四个进程、动作门控、视频、
   官方 status 和结果 JSON 全部正常；
2. 修复这次真实运行暴露的问题，保持单 instance 入口稳定；
3. 仿照 `scripts/Libero_eval/` 增加 instance 列表、断点续跑、批量汇总和失败重试；
4. 扩展到 BEHAVIOR 的更多任务；
5. 接入第二种人形机器人或第二种 VLA 后，再根据重复的真实实现提取通用接口；
6. 单独评估 LightNav 的长期归属；
7. 等人形链路覆盖现有能力后，再讨论移除 LIBERO 遗留代码。

每一步都要求运行入口使用新代码。禁止先创建空目录、单纯转发旧函数的包装类、
一组未被运行链路使用的抽象接口，或为一个 robot + policy + benchmark 组合创建
专用 `factory.py`。
