# Experiment protocol

每次正式实验必须先登记，至少记录：

| 字段 | 要求 |
| --- | --- |
| EXP_ID | 唯一标识 |
| data_version | 数据来源、aligned/unaligned 状态、划分和预处理版本 |
| model_version | 模型与结构版本；smoke-test 不算正式模型 |
| code_commit | 实际提交哈希及未提交变更 |
| config | base、task、local 合并后配置或版本化副本（移除机器敏感值） |
| random_seed | 全部随机源和缺失生成 seed |
| device | 实际设备与环境 |
| start_time / end_time | 含时区的真实时间 |
| output_path | 配置解析后的输出位置 |
| metrics | 真实完整评估集指标，运行前为 null |
| status | planned/running/completed/failed/cancelled |
| notes | 假设、错误、偏差和限制 |

planned 阶段的时间和指标为 null；失败如实记录，不能填写预期结果。raw 数据只读，专项无标签测试集不得用于调参、阈值、早停或模型选择。单元测试不是实验，不登记正式结果。

统一记录 Accuracy、macro-F1、weighted-F1、MAE、Pearson；F1 正式 averaging 尚未确认。F1/Pearson 应在完整评估集汇总后计算。Pearson 未定义时记录 null 和原因。

Q2 额外记录各模态原生时间轴上的 unit、position、duration、实际 start/end/length、seed、缺失率分母和事件元数据。不得将 element-wise dropout 报告为连续缺失。Q3 额外记录 AttributionTarget、方法、基线、梯度设置、重复 forward、时间证据映射。native diagnostics 和 attention 不得作为完整解释结果。

aligned/unaligned 正式路线必须引用 D-E-AUDIT-Q2-001 与 EXP-E-BACKBONE-ROUTING-001。当前无正式实验结果。
