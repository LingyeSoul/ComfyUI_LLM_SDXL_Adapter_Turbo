# ComfyUI LLM SDXL Adapter Turbo

## 本项目仅保留本项目描述，原始描述和更多说明请查看[ComfyUI_LLM_SDXL_Adapter](https://github.com/NeuroSenko/ComfyUI_LLM_SDXL_Adapter)

## 说明
1. 本项目需要转换后的rouwei Gemma Adapter的safetensors文件，请勿使用原版！！！  
2. 建议使用flash attn2加速推理，强烈建议安装  
3. 本项目推荐使用flash atten2 其他加速方式未测试，不保证效果，本项目没有严格测试，可能会有问题
4. 为避免与原版节点冲突，Turbo 版现在默认使用独立节点ID（例如 `TurboLLMModelLoader`），可与原版同时安装。
5. 如需兼容旧 Turbo 工作流（旧ID），可设置环境变量 `LLM_SDXL_TURBO_ENABLE_LEGACY_NODE_IDS=1`，但这会在与原版共存时重新引入覆盖风险。

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
