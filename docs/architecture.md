# Architecture

公共数据流：

```text
trusted PKL -> Adapter -> SampleSchema -> Dataset -> Collate -> Model
                                                    |
                                            independent T_text,
                                            T_audio, T_vision
```

依赖方向为任务层指向公共层。common.schema 定义 ModalitySequence/SampleSchema；data 提供 Dataset、Collate、Adapter；models 定义公共 forward/ModelOutput；training 组合 Model、Criterion、Optimizer；evaluation 独立提供指标。公共层不导入 q1/q2/q3。

Q2/Q3 共享 Dataset、Collate、Model API、Trainer、Metrics。Q2 仅增加连续缺失生成、鲁棒模块和鲁棒性评价。Q3 仅增加 attribution、explainer 和 evidence mapper。Q1 可以将异步输入对齐，也可以保留原生时间轴；共享契约不强制数据模式。

forward 分别接收 `[B,T_text,D_text]`、`[B,T_audio,D_audio]`、`[B,T_vision,D_vision]`，valid_masks/missing_masks 是按模态映射。ModelOutput 的隐藏状态保留各自时间长度，temporal_scores 为按模态字典，auxiliary 为未来重建、可靠性、蒸馏、一致性和代理表征预留。没有公共 `[B,T,3]` 掩码或时间分数假设。

## Prediction != Attribution

Trainer.predict 使用 no_grad，适合普通预测。AttributionMethod 直接访问 model、batch、AttributionTarget 和可选已有 output；需要梯度的方法在自身作用域使用 torch.enable_grad 并可重复 forward。因此归因不能通过 Trainer.predict 实现。

return_native_diagnostics 只请求模型自然产生的 attention、gate、reliability 等诊断量。return_native_diagnostics != full explainability，attention 也不是完整解释。正式归因方法尚未实现。

Trainer 不选择损失函数。Criterion 返回总损失和分项；SmokeTestMultiTaskCriterion 仅验证软件链路。任务辅助损失应由未来 Criterion 组合，而不是写进 Trainer 或 TaskHook。

当前 core 同时接受 aligned 和 unaligned，但正式主线需由 D-E-AUDIT-Q2-001 与 EXP-E-BACKBONE-ROUTING-001 决定。当前无正式 Transformer、EBMC、HME、LNLN、TFR 等 backbone，无正式实验结果。
