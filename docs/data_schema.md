# Data schema

ModalitySequence 表示一个模态的原生时间轴：

| 字段 | 形状 | 语义 |
| --- | --- | --- |
| features | `[T_m,D_m]` | 浮点特征 |
| valid_mask | `[T_m]` 或 None | True 表示真实位置 |
| missing_mask | `[T_m]` 或 None | True 表示有效位置上的不可用模态 |

None 在 collate 时分别物化为全 True valid_mask 和全 False missing_mask。missing_mask 不允许覆盖 valid_mask=False 的位置。

SampleSchema 包含 text/audio/vision 三个 ModalitySequence、id、可选 class_label/reg_label、raw_text 和 metadata。is_aligned 只比较三个 length，不改变处理路径。标签可为 Python 标量、单元素 Tensor 或 None；三分类编码目前为 0/1/2，实际语义与阈值等 D-E-AUDIT-Q2-001 确认。

Collate 独立 pad 每种模态：

```text
text:   [B,T_text_max,D_text]
audio:  [B,T_audio_max,D_audio]
vision: [B,T_vision_max,D_vision]

valid_masks[name]:   [B,T_name_max]
missing_masks[name]: [B,T_name_max]
```

padding 对两种 mask 都是 False，不能把 padding 当作真实缺失。全无标签 batch 的 class_label/reg_label 为 None；混合 batch 使用 label mask 排除占位值。

Dataset 不解析文件。MoseiPickleAdapter 只提供可信 pickle 的结构检查入口，实际字段到 SampleSchema 的映射在审计前故意未实现。aligned 与 unaligned 都通过同一 Schema/Collate/Model API。
