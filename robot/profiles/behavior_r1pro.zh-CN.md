# BEHAVIOR R1Pro 机器人构型

<!--
英文原文：robot/profiles/behavior_r1pro.md
英文原文 SHA-256：0ca1b2f5a87894a3cdb0bfe95e90731b4b8513b54af7c39f144d2cb60f735a00
修改英文原文时，请同步修订本译文、更新上述摘要，并运行
python -m unittest scripts.Behavior_eval.checks.test_profile_translation
运行时 Agent 仍使用英文原文；本文件供中文阅读。
-->

驱动：`behavior_r1pro`

BEHAVIOR 官方评测器负责仿真时间、重置、终止条件、评分和视频。
驱动通过评测器的 WebSocket 连接，按次授予有限的动作执行步数。
Agent 推理期间，仿真时间暂停。

## 坐标系与状态约定

以下动作使用的所有实测位置都必须位于 `robot_base` 坐标系，即
`artifacts/observations/observation.json` 中指定的坐标系。在该坐标系中，
+X 指向前方，+Y 指向左方，+Z 指向上方。绝不能直接把图像像素坐标当作公制动作坐标。

观测清单包含三个已标定的 RGB-D 视角（`head`、`left_wrist` 和
`right_wrist`）、官方全分辨率图像的精确相机内参、以米为单位的
`depth_linear` 深度数组、相对于机器人底盘的相机坐标变换，以及结构化的
R1Pro 本体感知状态。`ROBOT_STATE.md` 中的 `robots.behavior_r1pro`
包含实时的评测器和网关状态。每次物理动作后，都要重新读取这两个文件。

## 支持的动作

### `vla_execute`

调用冠军 PI_BEHAVIOR 策略；该策略可以控制全身运动和依赖视觉精度的接触操作。

本评测入口使用官方的当前任务指令。每次调用 `vla_execute` 时，都必须将
`robots.behavior_r1pro.effective_prompt` 原样复制为指令；当前局部阶段应写在
`PLAN.md` 和动作推理说明中，而不是改写策略提示词。`step` 是正整数的策略动作
执行预算，不得超过 `robots.behavior_r1pro.max_action_steps`。应分成有界的短段
执行，并在每段之间检查进展。

### `move_to_pose`

将一个末端执行器移动到可到达的实测位置附近：

```json
{"side": "left", "position_m": [0.55, 0.18, 0.92]}
```

`side` 为 `left` 或 `right`。`position_m` 必须来自 `object_location`
在 `robot_base` 坐标系下成功返回的结果，或者是根据该结果推导出的避障位置。
如果结果提供了目标物体或控件的边界框，应优先使用确定性的接触前目标形式：

```json
{
  "side": "auto",
  "bbox_world_min_m": [0.48, -0.12, 0.82],
  "bbox_world_max_m": [0.62, 0.02, 0.96],
  "standoff_m": 0.12
}
```

当 `side` 为 `auto` 时，驱动会选择距离目标更近的手臂，找到朝向该手臂的
边界框表面，并在距表面 8–15 厘米的位置停止。这样可避免把被物体占据的中心点
当作末端执行器的目标。显式提供 `position_m` 时，`side` 必须为 `left` 或
`right`。`orientation_quat` 使用 XYZW 顺序。省略该字段时，两种运动原语
都会保持当前手腕姿态。

### `move_linear`

沿笛卡尔直线，将一个末端执行器移动到可到达的实测位置，或移动指定的相对位移：

```json
{"side": "right", "delta_m": [0.0, 0.0, 0.08]}
```

仅在短距离的最后接近、抬起或沿可见空旷路径撤离时使用。`position_m` 绝对位置
与 `delta_m` 相对位移必须且只能提供其中一个；`orientation_quat` 可选。
这与通用 `attach` 技能的约定一致。

### `set_gripper`

```json
{"side": "left", "command": "open", "steps": 15}
```

`command`（`open` 或 `close`）与 `opening_m`（0.0–0.1 米）必须且只能提供
其中一个。`steps` 可选。只有当前计划和可见几何信息给出了具体理由时，才应
闭合夹爪。

### `hold_position`

```json
{"step": 1}
```

保持当前关节与夹爪位置，并将底盘速度设为零。仅用于同步或诊断。
Agent 推理本身不需要这个动作，因为动作之间仿真器已暂停。

### `set_control_mode`

每轮任务开始时处于全身 `navigation` 模式：

```json
{"mode": "navigation"}
```

当 Agent 判断下一个动作不应包含底盘平移时，可以在动作边界切换：

```json
{"mode": "manipulation"}
```

在 `manipulation` 模式下，网关会将官方 23 维策略动作的前三维（底盘 x/y
平移速度与偏航角速度）强制置零；躯干、双臂和双夹爪仍由 OpenPI 策略控制。
切换模式会重置 OpenPI，使下一次 VLA 调用根据当前观测重新规划。切换后应检查
`ROBOT_STATE.md` 中的 `control_mode` 与 `base_motion_enabled`。切回
`navigation` 模式会恢复底盘控制。两种模式都不要求在预设的任务阶段切换。

`gateway_status` 仅供读取。`attach` 是组合受支持的 `move_to_pose`、
`move_linear` 和 `set_gripper` 动作的通用 EMERGE 技能，不是独立的驱动动作。
该驱动尚未实现 `follow_arc` 和 `navigate_vln`。

每个本地运动原语都受到官方关节限位和单步关节变化上限约束。下发动作前，驱动会
将 URDF 正向运动学计算结果与观测到的末端执行器位置进行比较；若坐标约定不一致，
则拒绝执行。切回 OpenPI 策略时，系统会重置其动作块，使策略根据本地动作后的
新观测重新规划。

## 结果与成功判定

动作结果中的 `step_completed` 只表示指定的动作步数已经执行完毕，并与仿真器
状态完成同步；它不表示任务成功。唯一权威的成功信号是
`evaluation.success == true` 且 `evaluation.phase == "completed"`。
当 `evaluation.done` 为 true 时，应立即停止下发动作。
`evaluation.phase == "error"` 表示评测器故障，而非任务失败。任务运行期间，
`evaluation.steps` 尚不可用；`runtime_progress.gateway_action_steps` 仅是实时诊断
计数，不是官方评分。判断最近一次动作的结果前，应检查 `state_synced`。
