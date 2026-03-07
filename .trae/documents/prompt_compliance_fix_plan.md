# 提示词服从度下降问题分析与修复计划

## 问题概述

用户反馈提示词服从度大幅度下降，非常严重。通过对比当前版本与 commit 77c1e6cd6a6c378e7a9b545d573bd3988787cc99 原版代码，发现以下关键变更可能导致问题。

---

## 关键变更分析

### 1. **数据类型处理问题** (高风险)

**文件**: `llm_text_encoder.py`

**原版 (77c1e6c)**:
```python
hidden_states = outputs['hidden_states'][-1][:, skip_first:, :].to(torch.float)
```

**当前版本**:
```python
hidden_states = outputs['hidden_states'][-1][:, skip_first:, :].contiguous()
```

**问题**: 
- 原版使用 `.to(torch.float)` 强制转换为 float32
- 当前版本使用 `.contiguous()` 保持原有数据类型（可能是 bfloat16）
- 数据类型不一致可能导致适配器处理精度下降，影响提示词理解

---

### 2. **适配器架构重大变更** (高风险)

**文件**: `llm_to_sdxl_adapter.py`

**原版**: 使用 `nn.MultiheadAttention` 标准实现
**当前版本**: 使用 Flash Attention (`F.scaled_dot_product_attention`) + 分离的 QKV 投影

**具体问题**:

#### 2.1 Attention 机制变更
- **原版**: `nn.MultiheadAttention` 使用 `key_padding_mask`
- **当前版**: `F.scaled_dot_product_attention` 使用 `attn_mask`

#### 2.2 掩码处理差异
**原版**:
```python
key_padding_mask = ~mask.bool()  # True = 忽略此token
```

**当前版**:
```python
attn_mask = mask.unsqueeze(1).unsqueeze(2).bool()  # True = 有效
```

**问题**: 掩码逻辑反转可能导致 padding token 被错误处理

#### 2.3 Gemma 预设参数变更
**原版**:
```python
"gemma": {
    "n_wide_blocks": 3,
    "dropout": 0,
}
```

**当前版**:
```python
"gemma": {
    "n_wide_blocks": 2,
    "dropout": 0.1,
}
```

**问题**: 
- Wide blocks 从 3 减少到 2
- Dropout 从 0 增加到 0.1
- 架构变浅且引入随机性，可能降低提示词理解能力

---

### 3. **模型加载器 Attention 后端变更** (中风险)

**文件**: `llm_model_loader.py`, `llm_gguf_model_loader.py`

**新增功能**: 自动检测并启用 Flash Attention 2 或 SDPA

**问题**:
- Flash Attention 可能改变 hidden states 的数值特性
- 适配器权重可能是在标准 attention 下训练的，与 Flash Attention 不兼容

---

### 4. **T5Gemma 文本编码器变更** (中风险)

**文件**: `t5gemma_text_encoder.py`

**变更**: 添加了 `<eos>` token

```python
inputs = llm_tokenizer(
    text + "<eos>",  # 添加了 eos token
    ...
)
```

**问题**: 改变了 tokenization 行为，可能影响提示词编码

---

## 修复方案

### 方案一: 恢复关键数据类型处理 (推荐优先)

**修改文件**: `llm_text_encoder.py`

```python
# 当前代码
hidden_states = outputs['hidden_states'][-1][:, skip_first:, :].contiguous()

# 修复为
hidden_states = outputs['hidden_states'][-1][:, skip_first:, :].to(torch.float).contiguous()
```

**预期效果**: 恢复数据类型一致性，提高数值精度

---

### 方案二: 修复掩码处理逻辑

**修改文件**: `llm_to_sdxl_adapter.py`

在 `TransformerBlock.forward` 中:
```python
# 当前代码 (可能有问题)
if mask is not None:
    attn_mask = mask.unsqueeze(1).unsqueeze(2).bool()

# 修复为 - 反转掩码逻辑
if mask is not None:
    # mask: 1 = 有效, 0 = padding
    # scaled_dot_product_attention: True = 参与计算, False = 忽略
    attn_mask = mask.unsqueeze(1).unsqueeze(2).bool()
```

需要验证掩码逻辑是否正确处理 padding

---

### 方案三: 恢复 Gemma 预设参数

**修改文件**: `llm_adapter_loader.py`

```python
"gemma": {
    "llm_dim": 1152,
    "sdxl_seq_dim": 2048,
    "sdxl_pooled_dim": 1280,
    "target_seq_len": 308,
    "n_wide_blocks": 3,  # 从 2 恢复为 3
    "n_narrow_blocks": 3,
    "num_heads": 16,
    "dropout": 0,  # 从 0.1 恢复为 0
}
```

---

### 方案四: 提供 Attention 后端选择

**修改文件**: `llm_model_loader.py`

确保用户可以选择使用 eager attention 而非强制使用 Flash Attention

---

## 实施计划

### 阶段 1: 快速修复 (预计 1-2 小时)

1. **修复数据类型处理** - `llm_text_encoder.py`
   - 添加 `.to(torch.float)` 转换
   - 这是最简单的修复，可能解决大部分问题

2. **验证掩码逻辑** - `llm_to_sdxl_adapter.py`
   - 检查 `attn_mask` 是否正确处理
   - 确保 padding token 被正确忽略

### 阶段 2: 架构参数恢复 (预计 1 小时)

3. **恢复 Gemma 预设参数** - `llm_adapter_loader.py`
   - 将 `n_wide_blocks` 从 2 改回 3
   - 将 `dropout` 从 0.1 改回 0

### 阶段 3: 测试验证 (预计 2-3 小时)

4. **功能测试**
   - 使用标准提示词测试图像生成质量
   - 对比修复前后的提示词服从度

5. **回归测试**
   - 确保其他功能正常工作
   - 检查内存使用和性能

---

## 文件修改清单

| 优先级 | 文件 | 修改内容 |
|--------|------|----------|
| P0 | `llm_text_encoder.py` | 恢复 `.to(torch.float)` 数据类型转换 |
| P0 | `llm_to_sdxl_adapter.py` | 验证并修复掩码处理逻辑 |
| P1 | `llm_adapter_loader.py` | 恢复 Gemma 预设参数 (n_wide_blocks=3, dropout=0) |
| P2 | `llm_model_loader.py` | 确保 attention 后端可配置 |

---

## 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 修复后仍有问题 | 中 | 高 | 逐步应用修复，逐个验证 |
| 引入新问题 | 低 | 中 | 充分测试后再合并 |
| 性能下降 | 低 | 低 | 监控推理时间 |

---

## 测试建议

1. **使用标准测试提示词**:
   - "1girl, red hair, blue eyes, standing in a garden"
   - "masterpiece, best quality, detailed landscape"

2. **对比指标**:
   - 提示词中关键元素的出现率
   - 图像与提示词的语义匹配度
   - 生成图像的质量评分

3. **多轮测试**:
   - 使用不同随机种子测试稳定性
   - 测试不同复杂度提示词
