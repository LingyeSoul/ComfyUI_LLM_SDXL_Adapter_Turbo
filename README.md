# ComfyUI LLM SDXL Adapter Turbo

## 本项目仅保留本项目描述，原始描述和更多说明请查看[ComfyUI_LLM_SDXL_Adapter](https://github.com/NeuroSenko/ComfyUI_LLM_SDXL_Adapter)

## 说明
1. 本项目需要转换后的rouwei Gemma Adapter的safetensors文件，请勿使用原版！！！  
2. 建议使用flash attn2加速推理，强烈建议安装  
3. 本项目推荐使用flash atten2 其他加速方式未测试，不保证效果，本项目没有严格测试，可能会有问题
4. 为避免与原版节点冲突，Turbo 版现在默认使用独立节点ID（例如 `TurboLLMModelLoader`），可与原版同时安装。
5. 如需兼容旧 Turbo 工作流（旧ID），可设置环境变量 `LLM_SDXL_TURBO_ENABLE_LEGACY_NODE_IDS=1`，但这会在与原版共存时重新引入覆盖风险。
6. `Apply LLM To SDXL Adapter` 节点新增 `enable_diagnostics` 参数（默认关闭）；建议仅在排查质量问题时开启，日常推理保持关闭以避免额外同步开销。

## 感谢原作者

## 🎯 rouwei Gemma Adapter

**下载链接：**
- [HuggingFace 仓库](https://huggingface.co/lingyesoul/rouweiGemmaAdapter_converted)

**自行转换：**
- 适用于已经下载了原版adapter的用户
- 运行 `python convert_adapter_format.py 原始adapter路径`转换格式
- 运行 `python verify_adapter.py 转换的adapter路径`进行验证

## 📦 安装
### 安装节点
1. 将仓库克隆到 `ComfyUI/custom_nodes/`：
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/LingyeSoul/ComfyUI_LLM_SDXL_Adapter_Turbo.git
```
3. 安装flash attn2
这有助于加速推理，强烈建议安装  

2. 重启 ComfyUI

### 设置 RouWei-Gemma 适配器

1. **下载适配器：**
   - [HuggingFace 仓库](https://huggingface.co/lingyesoul/rouweiGemmaAdapter_converted) 下载
   - 将适配器文件放入 `ComfyUI/models/llm_adapters/`


### 提示
本项目目前仅支持gemma-3-1b-it模型，建议使用原始非量化模型，如果使用GGUF，加载器会像原版一样强制加载原始tokenizer等，以保证GGUF模型不会过多损失效果

> GGUF 加载器为离线优先：不会自动联网下载 tokenizer。
> 请确保 `ComfyUI/models/llm/gemma-3-1b-it/` 下至少存在 `tokenizer_config.json`、`special_tokens_map.json`、`tokenizer.json`。

## 🧩 条件处理节点说明（Conditioning）（本部分由AI生成，作者不保证解释合理）

送进采样链路（KSampler 等）的 `conditioning` 需要与 SDXL 侧语义一致，主要包括三部分：

- **形状语义**：主条件张量是 `[batch, tokens, channels]`，可被采样器和后续节点直接消费。
- **元数据语义**：`pooled_output`、`width/height`、`target_width/target_height`、`crop_w/crop_h` 等信息需要可用且连贯。
- **数值类型/设备语义**：实际进入采样链路的 conditioning 应保持 dtype/device 一致，避免拼接或后处理阶段出现隐式转换问题。

### 1) Apply 节点中的 dtype 对齐是“前向稳定性”措施

在 `Apply LLM To SDXL Adapter` 与 `Apply T5Gemma LLM to Adapter` 中，输入 hidden states 会先对齐到 adapter 的 device/dtype 后再做前向。这一步的目的主要是保证 adapter 算子执行稳定，并不直接定义最终对外 conditioning 的统一格式。

换句话说，这里解决的是“**能稳定算出来**”，不是“**最终一定与另一条 conditioning 完全一致**”。

### 2) Combine/Concat 节点负责“最终一致性对齐”

在 `LLM Conditioning Combine` 和 `LLM Conditioning Concat` 中，合并逻辑会以第一路 conditioning（`conditioning_1` / `conditioning_to`）的 dtype/device 作为目标，把另一路强制对齐后再拼接。

这意味着：

- 只要最终进入采样链路前走了 `combine/concat`，dtype/device 一致性路径就是有保障的。
- “编码器直接按 adapter dtype 输出”最多属于中间优化，不能替代最终合并节点的对齐职责。

### 3) token 截断策略与 disable 选项

`LLM Conditioning Combine` 与 `LLM Conditioning Concat` 提供以下 `truncate_strategy`：

- `balanced`：两路尽量均衡保留（默认）。
- `keep_start`：保留前部 token。
- `keep_end`：保留尾部 token。
- `disable`：**不截断**，忽略 `max_tokens` 上限，保留完整拼接序列。

## 📁 文件结构示例

```
ComfyUI/models/
├── llm/gemma-3-1b-it/
│   ├── added_tokens.json
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   ├── tokenizer.model
│   └── tokenizer_config.json
├── llm_adapters/
│   └── rouweiGemmaAdapter_converted.safetensors
```

## 🔍 调试

要启用详细日志记录，编辑 `__init__.py`：
```python
# 从：
logger.setLevel(logging.WARN)
# 改为：
logger.setLevel(logging.INFO)
```
