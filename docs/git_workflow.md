# Git workflow

约定分支：

| 分支 | 用途 |
| --- | --- |
| main | 稳定版本，禁止直接开发 |
| develop | 共享集成分支 |
| feature/q1 | Q1 提取、对齐、导出 |
| feature/q2 | Q2 专属模块 |
| feature/q3 | Q3 专属模块 |

Q2/Q3 共用 Dataset、Collate、Adapter、Model API、Trainer 和 Metrics，不得各自复制。对 common/data/models/training/evaluation 的公共修改必须先同步并基于 develop 集成；公共接口变更在合并前通知另一分支，写清字段、形状、独立时间轴、掩码和兼容影响。

feature/q2 的业务实现限于 q2 目录及相关配置/测试/文档；feature/q3 同理。正式数据路线从 D-E-AUDIT-Q2-001 和 EXP-E-BACKBONE-ROUTING-001 的真实证据决定，分支不得自行将 aligned 或 unaligned 写成正式默认。

合并前运行 CPU compileall 与 pytest，检查未提交数据、权重、日志、机器配置和虚假结果。不得删除或重建项目 .git，不自动 commit/push，不创建嵌套仓库。
