# e-multimodal-sentiment

中国研究生数学建模竞赛 E 题的本地 Python/PyTorch 探索工程。项目研究文本、语音、视觉三模态情感预测，采用 src-layout，支持 Python 3.10+。当前没有正式模型，也没有正式实验结果。

## 任务和共享核心

- Q1：原始视频的特征提取、时序对齐和导出。接口不会自动调用或下载 BERT、wav2vec2、CLIP。
- Q2：局部连续模态缺失下的三分类、强度回归、鲁棒性评价和附件 3 推理。
- Q3：完整三模态预测、归因接口、时间重要性和证据映射，面向附件 4。

Q2/Q3 共用 Dataset、Collate、Model API、Trainer 和 Metrics。Q2 专属 missing generator、robust modules、robustness eval；Q3 专属 attribution、explainer、evidence mapper。公共模块不导入任务模块，Q2/Q3 不复制共享核心。

公共契约同时支持 aligned（T_text = T_audio = T_vision）和 unaligned（各时间长度可以不同）。这只是工程兼容能力，不表示已选择正式数据主线。真实选择必须经过 D-E-AUDIT-Q2-001 数据审计和 EXP-E-BACKBONE-ROUTING-001 路由实验。

唯一可运行的 SmokeTestModel 只验证接口和训练链路，不是正式比赛模型或基线。native diagnostics 是模型天然产生的 gate、attention 或 reliability 等诊断量；attention 不等于完整解释，诊断量也不能替代归因实验。

## 环境和配置

项目代码不下载数据或预训练模型。已有依赖环境可运行：

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

base.yaml 只放共享配置，q1/q2/q3.yaml 只放任务配置。每台机器复制：

```text
configs/local.example.yaml
→ configs/local.yaml
```

然后只在 local.yaml 中填写机器路径和设备。该文件被 Git 忽略；示例不含真实绝对路径。配置加载顺序可写成 `load_config(base, task, local)`，后面的显式覆盖前面的值。

## 数据契约

每个 SampleSchema 包含三个 ModalitySequence；每个模态独立保存：

- features：`[T_m, D_m]`
- valid_mask：可选布尔 `[T_m]`，表示真实位置
- missing_mask：可选布尔 `[T_m]`，表示有效位置上的模态缺失

missing_mask 不能覆盖 invalid/padding。数值为零不会自动推断为缺失。Collate 分别 pad 三条时间轴，batch 的特征是 `[B,T_m_max,D_m]`，掩码是 `[B,T_m_max]`；padding 在两类掩码中都为 False。无标签专项 batch 的 class_label/reg_label 为 None，并保留标签存在掩码。

Dataset 只保存 SampleSchema，不解析 pickle。数据流为 PKL → Adapter → SampleSchema → Dataset → Collate → Model。真实 MOSEI 字段映射留待 D-E-AUDIT-Q2-001，不猜测未知字段。

## 推理、归因和损失

Trainer.predict 是普通 `torch.no_grad()` 推理。Prediction != Attribution：未来的 Integrated Gradients、ablation 等归因方法直接接收 model、batch 和 target，可自行用 `torch.enable_grad()` 与重复 forward，不经过 Trainer.predict。

Trainer 只负责 forward、调用 Criterion、backward 和 optimizer.step。CE、MSE、任务权重、重建或蒸馏损失均属于 Criterion。SmokeTestMultiTaskCriterion 的 CE+MSE 只用于测试，不代表正式损失。

Metrics 支持 Accuracy、macro-F1、weighted-F1、MAE、Pearson。最终正式 F1 averaging 尚未确认，因此同时报告 macro 和 weighted；无标签专项集不参与调参或指标计算。

## 只读数据审计

```bash
python scripts/audit_data.py <可信pickle路径> --trusted-pickle --mode quick
python scripts/audit_data.py <可信pickle路径> --trusted-pickle --mode full
```

quick 为默认模式，只报告结构、字段、shape、dtype、样本数和标签 shape。full 增加 NaN、Inf、zero ratio、数值范围和标签范围；对大型 unaligned 文件可能需要大量内存和时间。脚本只以 rb 读取并向标准输出打印 JSON，不修改 raw 数据。pickle 能执行代码，只能检查可信来源。

数据、模型权重、输出和本地路径不进入 Git，raw 数据只读。实验必须按 [实验协议](docs/experiment_protocol.md) 登记后运行，不得用专项无标签测试集调参。外部代码来源记录见 [external_sources.md](docs/external_sources.md)，目前未引入候选实现。
