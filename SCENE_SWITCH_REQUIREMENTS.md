# TUI 场景切换（BDDL）需求与方案

状态：已实现场景切换，且 reset/scene 已提升为所有 robot driver 的统一接口；验证记录见文末。

## 已确定的需求

- 用户可在正在运行的 Emerge TUI 中切换 Controller 的 LIBERO 场景，无须重启 TUI 或 Controller。
- BaseDriver 强制要求实现 `reset_environment()`、`get_scene_catalog()`、`switch_scene(scene_id)` 和 `close()`；TUI/Controller 不再判断 LIBERO 或 BDDL，其他环境可以提供自己的 scene ID 和目录模型。
- LIBERO 配置使用 `libero.bddl_root` 指定浏览根目录，`libero.bddl_file_name` 指定相对该根目录的初始 BDDL 路径。加载时拼接两项。旧配置未设置根目录时，仍可使用原完整文件路径，并以文件所在目录为浏览根目录。
- `/scene` 从斜杠命令菜单进入，直接打开驱动提供的 scene catalog；方向键选择，Enter 进入由 label 形成的子目录或加载 scene，选择 `../` 返回上级。Esc 取消。LIBERO 从 BDDL 根目录开始，其他 driver 不受文件系统或 BDDL 约束。
- 切换前停止正在运行的 Agent，等待机器人动作取消确认；未确认时不切换、不清理旧任务。切换完成后新任务、机器人状态和观测必须对应新场景，旧动作不能在新环境执行。
- 切换场景要清空旧场景任务上下文并开启新会话，但保留会话和运行历史。清理范围与现有 `reset_workspace_context()` 一致：`PLAN.md`、`ROBOT_STATE.md`、`EMBODIED.md`、`memory/MEMORY.md`、动作队列、`artifacts/`。
- 用户选中场景并按 Enter 后，**这一次切换就要完成所有必要的重置工作**：停下旧任务和动作、清理旧场景任务上下文、重新创建环境并刷新状态和观测。无需另行输入 `/reset`，也不应先加载一次旧场景再加载新场景。TUI 命令处理应为 `/reset` 和 `/scene` 分别设置 `if/elif` 分支，场景切换单独实现对应流程，不通过调用 `/reset` 命令或 `reset_environment()` 来代替切换。
- 如果目标场景加载失败，明确报告 Controller 未就绪，由用户再选择场景重试；不自动回退到旧场景，也不自动重放动作。

本需求只定义**选择并切换的当次操作**；“之后再单独执行 `/reset` 应加载哪个 BDDL”不是此次讨论的前置条件，不把它与场景切换的流程混为一谈。

## 实现方案

1. **取得当前场景：**Controller 通过 `BaseDriver.get_scene_catalog()` 获取 driver-owned 的场景根名、当前 ID 和 entries，并写入工作区 `.controller/scene.json`；TUI 只读取这些通用字段。Controller 启动和每次加载时更新状态；加载失败时标记未就绪、清除当前 ID 但保留 catalog，以便重试。
2. **TUI 选择：**沿用 `prompt_toolkit` 的浮层交互，按层级列出子目录和 `.bddl` 文件，目录优先；进入目录不影响当前任务，只有选中文件才发起切换。支持 `../` 返回上级和文本筛选，Esc 关闭。当前项标记清楚；仅在环境就绪且选中当前文件时不发起切换。Controller 离线、非 LIBERO driver 或目录不可读时提示原因。空子目录仍可返回上级。
3. **一次完成切换与重置：**在 TUI 命令处理逻辑中为 `/scene` 添加独立分支和切换流程，分别执行安全停机与取消确认、Controller 关闭旧 driver、清理工作区和创建新会话。Controller 按操作类型分别构造新 driver 并调用 `reset_environment()` 或 `switch_scene(scene_id)`，不把切换请求转成 reset 请求。不能调用现有 `/reset` 作为切换实现；可以复用底层取消动作、工作区清理和文件协议工具。
4. **边界和失败：**Controller 接收请求后验证目标 ID 在当前 catalog 中，再交由 driver 做最终校验。切换期间阻止新任务和重复切换；未停稳时不清理；清理后加载失败则保持未就绪并报告错误，可再次使用 `/scene` 重试，不声称旧场景仍可用。切换成功后侧栏显示 driver 提供的当前 label。
5. **保留原有命令：**`/reset` 作为独立命令照常可用；新功能不要求改变其实现或使用方式。场景切换所需的停止、清理和重新加载在当次操作内完成，不依赖用户随后执行 `/reset`。

当前 BDDL 来自 Controller 的 `--driver-config`，TUI 并不持有这份配置。仅在 TUI 更改显示或直接调用已创建环境的 `reset()` 无法换场景。两个分支可以使用相同的底层工具，但切换场景有自己完整的控制流程和目标 BDDL 参数，不改变 `/reset` 的入口与现有流程。

## 验收要点

- `/scene` 从 driver catalog 开始，支持按 label 进入分组和返回上级；进入分组不触发切换，Enter 选中 scene 才切换，Esc 取消；无效 ID 不会清理任务。
- 运行中切换先取消 Agent/机器人动作；取消未确认时旧环境和任务文件不被清理。
- 切换成功后新场景产生新的机器人状态/观测，旧计划和动作队列消失，历史保留。
- 加载失败提示未就绪并允许重试；过期、重复或切换中请求不重复清理或加载。
- 一次 `/scene` 选择即可完成切换和必要的重置；原有 `/reset` 命令仍可单独使用。Controller 重启后，TUI 不显示上次进程的过期状态。
- 测试使用模拟 driver 覆盖上述协议和状态流转，无须依赖真实 MuJoCo。

## 范围说明

TUI/Controller 不解析环境类型，不自动回退。Controller 重启后的选中场景是否持久化暂不作为本功能要求。LIBERO 的实现仍限制在配置 BDDL 根目录内，Plus 可使用虚拟文件名，评测脚本可使用各自的数据目录。其他驱动只需实现 `robot/DRIVER_INTERFACE.md` 中的契约。

## 配置示例与评测

```json
{
  "libero": {
    "bddl_root": "third_party/openpi/third_party/libero/libero/libero/bddl_files",
    "bddl_file_name": "libero_object/pick_up_the_cream_cheese_and_place_it_in_the_basket.bddl"
  }
}
```

相对根目录按 Controller 的工作目录解析，与原路径配置一致。评测脚本按所选任务生成两项配置，基础评测 JSON 无需写死某个场景。Plus 保留虚拟文件名后缀；Pro 对符号链接组成的数据目录使用任务实际的数据源根目录，避免生成错误的相对路径。

## 使用和验证

更新代码后先重启一次 TUI 和 Controller。此后使用 `/` → `/scene` → Enter 进入 suite 子目录 → 选择 BDDL → Enter，即可在进程内完成场景切换。`../` 返回上级，Esc 取消，也可输入名称片段筛选。Controller 通过 `.controller/scene.json` 提供当前场景和根目录信息。

初版场景切换已通过 10 项模拟 driver 测试及 Ruff 检查，测试文件已按要求删除；尚未进行真实 MuJoCo 场景联调。根目录浏览和评测路径拆分的验证使用临时脚本，项目内不保留测试文件。

本次临时验证已通过：新旧配置路径解析、目录进入和返回、空目录导航、跨 suite 的 Enter 切换、根目录边界、原 reset 流程，以及标准 LIBERO / Plus 虚拟文件名 / Pro 原始与生成数据符号链接的评测配置生成。涉及的 Python 文件通过 Ruff 检查，未运行完整仿真评测。
