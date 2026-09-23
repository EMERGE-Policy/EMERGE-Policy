# LightNav-0 + SONIC 导航 Skill 文档索引

本目录记录从 EMERGE-Humanoid fork 初始化到 LightNav-0 + SONIC 导航 Skill
落地和实测的全过程。新协作者建议先阅读 integration 的
[主 README](../integrations/lightnav_sonic_emerge/README.md)，再根据任务查阅以下中文文档。

| 文档 | 内容 | 推荐读者 |
|---|---|---|
| [01_EMERGE_Policy论文与代码架构.md](01_EMERGE_Policy论文与代码架构.md) | EMERGE 论文、Controller、workspace、driver、agent 与 observation 架构 | 需要理解整体框架的开发者 |
| [02_环境配置.md](02_环境配置.md) | EMERGE 基础环境及可选组件安装 | 首次配置环境的开发者 |
| [03_LightNav0_SONIC导航Skill接入方案.md](03_LightNav0_SONIC导航Skill接入方案.md) | 接口选择、动作生命周期、安全约束和分阶段方案 | 继续设计或评审 integration 的开发者 |
| [04_导航Skill执行与测试终端指令.md](04_导航Skill执行与测试终端指令.md) | 离线测试及 ProcTHOR 六终端真实模型运行命令 | 复现实验的开发者 |
| [05_导航Skill实现文件与改动说明.md](05_导航Skill实现文件与改动说明.md) | 新增文件、实际调用链和验证边界 | Code review 与交接人员 |

当前已验证的是手动确定性入队后的真实链路：

```text
ACTION.md → EMERGE Controller → LightNav-0 → waypoint adapter
          → SONIC planner/WBC → ProcTHOR MuJoCo G1
```

Main Agent（如 Codex/API）到 `navigate_vln` 的规划入口已经具备，但尚未完成端到端
效果评估；真机运行也不在当前验证范围内。文档中的 `/home/embodied/...` 是本次实验机
路径，其他机器应优先使用主 README 中的可移植环境变量命令。
