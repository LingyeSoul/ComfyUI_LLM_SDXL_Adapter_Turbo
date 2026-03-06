# 代码修复计划

## 修复任务

### 1. llm_to_sdxl_adapter.py 修复

#### 1.1 修复 converted_blocks 重复计数问题 (L70)
- **问题**: `converted_blocks += 1` 在每个匹配的 key 都递增，应该按 block 计数
- **修复**: 使用 set 记录已处理的 base_path，避免重复计数

#### 1.2 提取重复代码为通用函数 (L73-119)
- **问题**: compression_attention 和 pooling_attention 转换逻辑重复
- **修复**: 创建 `_convert_attn_to_separate_qkv` 通用函数

#### 1.3 修复 seq_projection 检查方式 (L312)
- **问题**: 使用 `if self.seq_projection:` 可能会有问题
- **修复**: 改为 `if hasattr(self, 'seq_projection'):`

### 2. llm_adapter_loader.py 修复

#### 2.1 修复 checkpoint 属性检查 (L216-217)
- **问题**: `checkpoint` 是 dict，不会有 `input_norm` 属性
- **修复**: 改为 `any("input_norm" in k for k in checkpoint.keys())`

#### 2.2 清理未使用的变量 (L28)
- **问题**: `keys_to_remove` 定义但未使用
- **修复**: 删除该变量

#### 2.3 改进 forward hooks 清理 (L148-150)
- **问题**: 可能无法清理所有类型的 hooks
- **修复**: 遍历所有子模块清理 hooks

### 3. 代码优化

#### 3.1 添加类型注解
- 为关键函数添加类型提示

#### 3.2 改进注释
- 为新增的通用函数添加文档字符串
