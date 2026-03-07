"""
验证 adapter 文件格式转换
检查转换后的键名是否与新模型结构匹配
"""

import sys
import os

# 期望的新格式键名（基于 LLMToSDXLAdapter 模型结构）
EXPECTED_NEW_FORMAT_KEYS = {
    # Wide attention blocks (3 blocks with q/k/v/out proj)
    "wide_attention_blocks.0.q_proj.weight", "wide_attention_blocks.0.q_proj.bias",
    "wide_attention_blocks.0.k_proj.weight", "wide_attention_blocks.0.k_proj.bias",
    "wide_attention_blocks.0.v_proj.weight", "wide_attention_blocks.0.v_proj.bias",
    "wide_attention_blocks.0.out_proj.weight", "wide_attention_blocks.0.out_proj.bias",
    "wide_attention_blocks.1.q_proj.weight", "wide_attention_blocks.1.q_proj.bias",
    "wide_attention_blocks.1.k_proj.weight", "wide_attention_blocks.1.k_proj.bias",
    "wide_attention_blocks.1.v_proj.weight", "wide_attention_blocks.1.v_proj.bias",
    "wide_attention_blocks.1.out_proj.weight", "wide_attention_blocks.1.out_proj.bias",
    "wide_attention_blocks.2.q_proj.weight", "wide_attention_blocks.2.q_proj.bias",
    "wide_attention_blocks.2.k_proj.weight", "wide_attention_blocks.2.k_proj.bias",
    "wide_attention_blocks.2.v_proj.weight", "wide_attention_blocks.2.v_proj.bias",
    "wide_attention_blocks.2.out_proj.weight", "wide_attention_blocks.2.out_proj.bias",

    # Narrow attention blocks (3 blocks with q/k/v/out proj)
    "narrow_attention_blocks.0.q_proj.weight", "narrow_attention_blocks.0.q_proj.bias",
    "narrow_attention_blocks.0.k_proj.weight", "narrow_attention_blocks.0.k_proj.bias",
    "narrow_attention_blocks.0.v_proj.weight", "narrow_attention_blocks.0.v_proj.bias",
    "narrow_attention_blocks.0.out_proj.weight", "narrow_attention_blocks.0.out_proj.bias",
    "narrow_attention_blocks.1.q_proj.weight", "narrow_attention_blocks.1.q_proj.bias",
    "narrow_attention_blocks.1.k_proj.weight", "narrow_attention_blocks.1.k_proj.bias",
    "narrow_attention_blocks.1.v_proj.weight", "narrow_attention_blocks.1.v_proj.bias",
    "narrow_attention_blocks.1.out_proj.weight", "narrow_attention_blocks.1.out_proj.bias",
    "narrow_attention_blocks.2.q_proj.weight", "narrow_attention_blocks.2.q_proj.bias",
    "narrow_attention_blocks.2.k_proj.weight", "narrow_attention_blocks.2.k_proj.bias",
    "narrow_attention_blocks.2.v_proj.weight", "narrow_attention_blocks.2.v_proj.bias",
    "narrow_attention_blocks.2.out_proj.weight", "narrow_attention_blocks.2.out_proj.bias",

    # Compression attention
    "compression_q_proj.weight", "compression_q_proj.bias",
    "compression_k_proj.weight", "compression_k_proj.bias",
    "compression_v_proj.weight", "compression_v_proj.bias",
    "compression_out_proj.weight", "compression_out_proj.bias",

    # Pooling attention
    "pooling_q_proj.weight", "pooling_q_proj.bias",
    "pooling_k_proj.weight", "pooling_k_proj.bias",
    "pooling_v_proj.weight", "pooling_v_proj.bias",
    "pooling_out_proj.weight", "pooling_out_proj.bias",
}

# 旧格式的键名模式
OLD_FORMAT_PATTERNS = [
    ".attn.in_proj",      # e.g., wide_attention_blocks.0.attn.in_proj_weight
    ".attn.out_proj",     # e.g., wide_attention_blocks.0.attn.out_proj.weight
    "compression_attention.",  # e.g., compression_attention.q_proj.weight
    "pooling_attention.",      # e.g., pooling_attention.q_proj.weight
]


def analyze_adapter_keys(keys):
    """分析 adapter 文件的键名格式"""
    result = {
        'old_format_keys': [],
        'new_format_keys': [],
        'other_keys': [],
        'compression_pooling_old': [],
        'compression_pooling_new': [],
    }

    for key in keys:
        # 检查是否是旧格式
        is_old_format = any(pattern in key for pattern in OLD_FORMAT_PATTERNS)

        if is_old_format:
            result['old_format_keys'].append(key)
            if "compression_attention." in key or "pooling_attention." in key:
                result['compression_pooling_old'].append(key)
        elif key in EXPECTED_NEW_FORMAT_KEYS:
            result['new_format_keys'].append(key)
            if key.startswith("compression_") or key.startswith("pooling_"):
                result['compression_pooling_new'].append(key)
        else:
            result['other_keys'].append(key)

    return result


def verify_adapter_file(file_path):
    """验证 adapter 文件"""
    try:
        from safetensors.torch import load_file
        state_dict = load_file(file_path)
    except Exception as e:
        print(f"错误: 无法加载文件 {file_path}: {e}")
        return False

    keys = list(state_dict.keys())
    analysis = analyze_adapter_keys(keys)

    print(f"\n文件: {file_path}")
    print(f"总键数: {len(keys)}")
    print(f"旧格式键数: {len(analysis['old_format_keys'])}")
    print(f"新格式键数: {len(analysis['new_format_keys'])}")
    print(f"其他键数: {len(analysis['other_keys'])}")

    if analysis['old_format_keys']:
        print("\n旧格式键名 (需要转换):")
        for key in analysis['old_format_keys'][:10]:
            print(f"  - {key}")
        if len(analysis['old_format_keys']) > 10:
            print(f"  ... 还有 {len(analysis['old_format_keys']) - 10} 个")

    if analysis['compression_pooling_old']:
        print(f"\ncompression/pooling 旧格式键数: {len(analysis['compression_pooling_old'])}")
        for key in analysis['compression_pooling_old'][:5]:
            print(f"  - {key}")

    if analysis['compression_pooling_new']:
        print(f"\ncompression/pooling 新格式键数: {len(analysis['compression_pooling_new'])}")

    # 检查是否可以直接加载
    can_load_directly = len(analysis['old_format_keys']) == 0

    print("\n" + "="*50)
    if can_load_directly:
        print("✓ 此文件可以直接加载 (已经是新格式)")
    else:
        print("✗ 此文件需要转换才能加载")
        print(f"  需要转换的键数: {len(analysis['old_format_keys'])}")

    return can_load_directly


def main():
    if len(sys.argv) < 2:
        print("用法: python verify_adapter.py <adapter_file.safetensors>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在: {file_path}")
        sys.exit(1)

    verify_adapter_file(file_path)


if __name__ == "__main__":
    main()
