# AUDIT-E-Q2-TEXT-001：Q2 文本接口只读审计

**审计时间**：2026-09-23 21:39（Asia/Shanghai）。**结论：CASE-A（限本次固定源码、本地缓存权重与实测样本），无文本接口阻塞。** Attachment2 的原始 dense text 与用固定 MMSA `BertTextEncoder` 重算的表示在抽样中数值近乎一致；Attachment3 的 `text_bert` 可由同一 encoder 编为 `[B,50,768]`，并已通过当前 COMMON MulT 双头 wrapper 的一次真实 forward。此结论只证明接口和表示兼容，不是情感预测性能结论。

## 环境、版本与数据定位

| 项目 | 本次实际值 |
| --- | --- |
| 操作系统 | Windows 10, build 26200 |
| Python | `E:\miniconda3\envs\evigraph-radgraph\python.exe`，3.11.16 |
| PyTorch / transformers | `2.14.0+cpu` / `4.39.3`；实际运行设备 CPU |
| 项目 commit | `da6a99679a67e7357b72af0eba97ed6548667237` |
| MMSA commit | `a94e65d07fa1ae0d44e552390074b29b0898edfd`，与要求一致 |
| Attachment2 aligned | `D:\datasets\E题数据\附件2-数据集特征文件\aligned_50.pkl` |
| Attachment3 aligned | `D:\datasets\E题数据\附件3-模态缺失特征样本\对齐版本\附件3_01.pkl` 至 `D:\datasets\E题数据\附件3-模态缺失特征样本\对齐版本\附件3_30.pkl`，连续编号 01–30，共 30 个文件，全部逐文件读取 |
| Attachment3 另一候选 | `D:\datasets\E题数据\附件3-模态缺失特征样本\未对齐版本`，30 个 PKL；本次未作为 aligned 输入 |
| MMSA 源码根目录 | `D:\vs_pythonWks\external\MMSA`，只读 |

执行前先扫描数据根目录，再核对固定 MMSA commit；没有依赖历史聊天猜测数据文件位置。`huawei-e` 环境没有 PyTorch，故使用上表中已有的 CPU 环境。运行时设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，只使用 `D:\HuggingFace_Cache\hub\models--bert-base-uncased` 中已有缓存，未下载或安装内容。该环境缺少 `easydict`，审计脚本仅绕开 MMSA 顶层 `__init__` 的无关依赖导入，动态加载同一固定 commit 的**原始** encoder/MulT 源码；未复制或改写算法。

## MMSA 源码确认

以下均为【源码确认】，依据固定 commit 的 `src/MMSA/models/subNets/BertTextEncoder.py`、`src/MMSA/data_loader.py`、`src/MMSA/models/missingTask/TFR_NET/model.py`、`src/MMSA/models/singleTask/MULT.py` 和 `src/MMSA/config/config_regression.json`：

- `BertTextEncoder.forward(text)` 接收 `[B,3,T]`。`text[:,0,:]` 是 token IDs，转 `long`；`text[:,1,:]` 是 attention mask，转 `float`；`text[:,2,:]` 是 segment/token-type IDs，转 `long`。函数把三者送进 Hugging Face `BertModel`，返回 `[B,T,768]` 的最后隐层。该路径没有针对 `text_bert` 的 transpose/permute。
- 默认模型和 tokenizer 同为 `bert-base-uncased`；`use_finetune` 决定 encoder 前向是否求梯度。本次只读前向采用 `use_finetune=False` 和 `eval()`。TFR-Net 构造时采用 `args.use_bert_finetune`；其配置标为 `true`。本次没有运行 TFR-Net。
- MMSA loader 在 `use_bert=true` 时选 `text_bert.astype(np.float32)`，否则选 dense `text`；数据读取阶段不改变 `[N,3,50]` 布局。BERT 根据 attention mask 屏蔽 padding 的注意力参与；其返回张量仍覆盖 50 个位置，因此 padding 位置的 hidden 不应被假定为全零。
- TFR-Net 的 `forward` 接口分别是 `text=(原文本, 损坏文本, missing_mask_t)`、`audio=(原音频, 损坏音频, audio_mask, missing_mask_a)`、`vision=(原视觉, 损坏视觉, vision_mask, missing_mask_v)`；其中原文本和损坏文本都走同一个 `BertTextEncoder` 实例，`text[:,1,:]` 被取作 text mask。其内部 `missing_mask_*` **1 表示保留观测**，重构位置使用 `text_mask - missing_mask_t` 等差值。此命名不能直接等同于 COMMON Schema 中 1 表示人为缺失的 `missing_mask`。
- MMSA loader 的文本损坏逻辑保留首尾特殊 token，用 token ID `100`（BERT `[UNK]`）替代选中的内部 token，attention mask 保持原样；音频和视觉损坏通过置零实现。该 loader 注释称当前缺失任务仅支持 unaligned，不能据此认定原版 TFR-Net 已直接支持 Attachment3 aligned。
- MulT 在 `use_bert=false` 时接收 dense `[B,50,768]`；当前 COMMON wrapper 基于此路径。本次先独立运行相同固定源码的 `BertTextEncoder`，再把它的输出送入现有 bridge/wrapper，没有让 MulT 再编码一次。

## Attachment2 / Attachment3 真实 Schema

以下为【数据实测】。Attachment2 顶层为 `train/valid/test`，三个 split 字段均为 `raw_text, audio, vision, id, text_bert, classification_labels, regression_labels, text`。除 `raw_text` 字符串宽度不同外，结构一致。

| 来源 | 样本数 | text_bert 原始 shape / dtype | dense text | audio / vision |
| --- | ---: | --- | --- | --- |
| Attachment2 train | 3,395 | `[3395,3,50]` / int64 | `[3395,50,768]` / float32 | `[3395,50,74]`、`[3395,50,35]` / float64 |
| Attachment2 valid | 728 | `[728,3,50]` / int64 | `[728,50,768]` / float32 | 对应 `[N,50,74]`、`[N,50,35]` / float64 |
| Attachment2 test | 727 | `[727,3,50]` / int64 | `[727,50,768]` / float32 | 对应 `[N,50,74]`、`[N,50,35]` / float64 |
| Attachment3 aligned | 30 文件，每文件 1 条 | 每文件 `[1,3,50]` / float32 | **无 dense text** | 每文件 `[1,50,74]`、`[1,50,35]` / float32 |

Attachment3 每个文件顶层只有 `test`，其字段均且仅为 `text_bert, audio, vision`，**没有 ID 或标签**。所有 30 个文件的 schema 一致。Attachment2/3 的 text_bert 布局均为 `[样本,通道,时间]`，无需转换维度；仅需按 encoder 源码转换 token/segment 为整数。Attachment3 虽存 float32，三个通道的有限值均为整数值。三个通道都无 NaN/Inf。

| 数据 | token IDs 最小/最大/unique 数 | attention 值域 | segment 值域 | CLS/SEP |
| --- | --- | --- | --- | --- |
| A2 train | 0 / 29,824 / 7,572 | `{0,1}` | `{0}` | 3,395/3,395 均有 |
| A2 valid | 0 / 29,589 / 3,191 | `{0,1}` | `{0}` | 728/728 均有 |
| A2 test | 0 / 29,476 / 3,229 | `{0,1}` | `{0}` | 727/727 均有 |
| A3 aligned | 0 / 28,428 / 277 | `{0,1}` | `{0}` | 30/30 均有 |

【数据推断】两批数据的 channel 顺序、整数 token 值域、attention 值域、token-type 值域、CLS=`101` / SEP=`102`、长度 50 均兼容固定 `bert-base-uncased` encoder；不能仅凭 channel 数为 3 判定，以上逐项检查和下述真实 forward 共同支持这一判断。

## Padding 与内部缺口

以下以每条序列末尾显式 SEP=`102` 定义有效边界；SEP 后是尾部 padding，CLS 与 SEP 自身不计入内部 span。统计覆盖 Attachment2 全部 4,850 条和 Attachment3 全部 30 条。

| 来源 | CLS→SEP 长度 min / median / max | 尾部 padding min / median / max | 内部 ID=0、attention=0、三通道全零 | 内部 ID=100 |
| --- | --- | --- | --- | --- |
| A2 train | 3 / 22 / 50 | 0 / 28 / 47 | 0 条样本，0 段 | 0 条、0 段 |
| A2 valid | 3 / 23.5 / 50 | 0 / 26.5 / 47 | 0 条样本，0 段 | 0 条、0 段 |
| A2 test | 4 / 23 / 50 | 0 / 27 / 46 | 0 条样本，0 段 | 0 条、0 段 |
| A3 aligned | 8 / 22 / 50 | 0 / 28 / 42 | 0 条样本，0 段 | 27 条、95 段、131 token |

Attachment3 的 ID=100 内部 span 最长 4 token；长度分布为 1:71 段、2:16 段、3:4 段、4:4 段。起点覆盖内部多个位置（例如 1、5、14、21、31、48），不是单一尾部 padding。所有数据的 SEP 后均未检测到非零三通道值。

【数据实测】Attachment3 文本缺口的可见模式是**内部 token ID 被换为 100，attention 仍为 1，segment 仍为 0**；未出现内部 token ID 置 0、attention 置 0 或三个 channel 同时置 0。Attachment2 全部 split 没有该内部 100 模式。【数据推断】结合 MMSA loader 源码，100 是合理的文本损坏候选标记；但 Attachment3 没有原文本或官方缺失 mask，不能仅凭 100 在语义上唯一证明每个 token 都是 Q2 人工缺失。BERT 自然的 `[UNK]` 也使用 100，虽然本次 Attachment2 未观察到。因而可建立“候选缺口代理”，不应把它当作已确证的 COMMON `missing_mask` 真值；更不能用 `feature==0` 统一判定三模态 Q2 缺失。

## 同一固定 BertTextEncoder 真实前向

从 A2 `train` 取前 3 条、从 A3 对齐文件 `01–03` 各取 1 条。原始和 canonical 布局都为 `[3,3,50]`，使用**同一个**在本地缓存加载的固定 MMSA `BertTextEncoder` 实例分别前向；设备 CPU，`eval()` / `torch.no_grad()`。没有训练。

| 来源 | 输出 shape / dtype | min / max | mean / std | NaN / Inf |
| --- | --- | --- | --- | --- |
| A2 | `[3,50,768]` / float32 | −9.408875 / 3.869912 | −0.010189 / 0.463036 | 0 / 0 |
| A3 | `[3,50,768]` / float32 | −9.245178 / 3.926651 | −0.009397 / 0.447190 | 0 / 0 |

## Attachment2 dense 与重算 BERT 的数值对照

比较同一 3 条 A2 train 样本的 `text` 与 `BertTextEncoder(text_bert)`；二者均 `[3,50,768]`、float32。全序列 150 个 timestep：平均绝对差 **5.7958×10⁻⁷**，最大绝对差 **1.1921×10⁻⁵**，MSE **6.4332×10⁻¹³**；逐 timestep cosine 均值/中位数为 **1.0/1.0**，最小/最大约 **0.99999970/1.00000024**。样本 flatten cosine 为 **1.00000024、1.00000060、1.00000095**。

仅按 attention=1 纳入的 87 个 timestep：平均绝对差 **6.0437×10⁻⁷**，最大绝对差 **6.1989×10⁻⁶**，MSE **6.1816×10⁻¹³**；逐 timestep cosine 最小/最大约 **0.99999976/1.00000024**。dense 与重算编码的整体 mean/std 分别均约 **−0.010189/0.463036**；每 timestep norm 均值分别 **12.638600/12.638599**。全序列统计包含 padding 上的 BERT 输出，所以同时给出有效位置统计。浮点 cosine 略大于 1 属计算舍入。

【源码 + 实测综合判断】这批 A2 dense text 与该固定缓存 encoder 的生成结果可视为同一特征表示流程，至少上述真实样本数值近乎相同；CASE-A 成立。并未在全部 4,850 条样本上重复 BERT 编码，故不声称逐条完全相等。后续若更换预训练缓存、transformers 版本或启用 BERT finetune，应重新做一致性抽查。

## Attachment3 → COMMON MulT 最小真实前向

在前述 3 条 A3 样本上，将固定 encoder 输出与同样本 audio/vision 转为 float32，经过现有 `BatchSchema` → `batch_to_mmsa_mult_inputs` → 固定 MMSA `MULT` → COMMON `MMSAMulTMultiTask`。脚本复用现有 `scripts/smoke_mmsa_mult.py` 的固定 config/commit 构造函数；没有重写 MulT 或 Transformer。

| 张量 | 实际 shape | dtype | NaN / Inf |
| --- | --- | --- | --- |
| text | `[3,50,768]` | float32 | 0 / 0 |
| audio | `[3,50,74]` | float32 | 0 / 0 |
| vision | `[3,50,35]` | float32 | 0 / 0 |
| class_logits | `[3,3]` | float32 | 0 / 0 |
| regression | `[3]` | float32 | 0 / 0 |
| fused_hidden | `[3,180]` | float32 | 0 / 0 |

本次为了只验证接口，临时 BatchSchema 使用全 True 的 valid/observed 和全 False 的 missing；**这不是 Attachment3 缺失 mask 的识别结果**。MulT 双头是本次新建的未训练随机实例；没有 checkpoint、训练、指标、预测质量结论。Attachment3 无标签，没有计算 Accuracy、F1、MAE 或 Pearson。

## CASE 判定、剩余限制与建议

**CASE-A。** A2/A3 `text_bert` 布局与通道语义一致；同一 encoder 在两者上真实成功前向；A2 dense 与 encoder 输出的数值对照高度一致；A3 编码结果真实进入当前 COMMON MulT 并完成双头 forward。因此可继续维持 A2 train/valid dense text 与 A3 同一固定 encoder 的接入路径。**没有 `BLOCKER-E-Q2-TEXT-001` 文本接口阻塞。**

剩余问题只涉及后续缺失建模，不改变本次 CASE：A3 没有显式官方缺失 mask，token 100 只能作为文本缺口候选；音频/视觉置零也不能单凭 `feature==0` 判定人为缺失；原版 TFR-Net 的 loader 不是 aligned Attachment3 的直接适配器。正式训练或推理时须固定 encoder 权重/版本，显式维护 COMMON 四类 mask 的语义，并把这些不确定性写入 Q2 实验设计。本次不进入 Q2 模型开发。

## 本次文件与执行边界

新增 `scripts/audit_q2_text_interface.py`（只读审计脚本）及本报告 `AUDIT-E-Q2-TEXT-001.md`。执行生成的明细 JSON 暂存在被 Git 忽略的 `outputs/audit_q2_text_interface.json`，含 30 个文件的逐文件 schema 和完整统计。本次没有修改原始数据、MMSA、COMMON Schema/Backbone/Trainer、Q2/Q3，也没有训练、安装依赖、git commit 或 git push。
