"""
诊断 adapter 文件问题
"""
import sys
import os

def analyze_file(file_path):
    """分析 adapter 文件的键名"""
    try:
        from safetensors.torch import load_file
        state_dict = load_file(file_path)
    except Exception as e:
        print(f"错误: 无法加载文件 {file_path}: {e}")
        return

    keys = list(state_dict.keys())

    # 分类键名
    compression_attention_keys = [k for k in keys if "compression_attention." in k]
    pooling_attention_keys = [k for k in keys if "pooling_attention." in k]
    compression_q_proj_keys = [k for k in keys if "compression_q_proj." in k]
    pooling_q_proj_keys = [k for k in keys if "pooling_q_proj." in k]
    wide_attn_keys = [k for k in keys if "wide_attention_blocks." in k and ".attn." in k]
    narrow_attn_keys = [k for k in keys if "narrow_attention_blocks." in k and ".attn." in k]

    print(f"\n文件: {file_path}")
    print(f"总键数: {len(keys)}")

    print("\n" + "="*60)
    print("Compression Attention:")
    if compression_attention_keys:
        print(f"  旧格式键数: {len(compression_attention_keys)}")
        for k in compression_attention_keys[:4]:
            print(f"    - {k}")
        if len(compression_attention_keys) > 4:
            print(f"    ... 还有 {len(compression_attention_keys) - 4} 个")
    else:
        print("  没有旧格式键 (compression_attention.)")

    if compression_q_proj_keys:
        print(f"  新格式键数: {len(compression_q_proj_keys)}")
        for k in compression_q_proj_keys[:4]:
            print(f"    - {k}")
    else:
        print("  没有新格式键 (compression_q_proj.)")

    print("\n" + "="*60)
    print("Pooling Attention:")
    if pooling_attention_keys:
        print(f"  旧格式键数: {len(pooling_attention_keys)}")
        for k in pooling_attention_keys[:4]:
            print(f"    - {k}")
        if len(pooling_attention_keys) > 4:
            print(f"    ... 还有 {len(pooling_attention_keys) - 4} 个")
    else:
        print("  没有旧格式键 (pooling_attention.)")

    if pooling_q_proj_keys:
        print(f"  新格式键数: {len(pooling_q_proj_keys)}")
        for k in pooling_q_proj_keys[:4]:
            print(f"    - {k}")
    else:
        print("  没有新格式键 (pooling_q_proj.)")

    print("\n" + "="*60)
    print("Wide Attention Blocks (带 .attn. 的键):")
    if wide_attn_keys:
        print(f"  旧格式键数: {len(wide_attn_keys)}")
        for k in wide_attn_keys[:4]:
            print(f"    - {k}")
        if len(wide_attn_keys) > 4:
            print(f"    ... 还有 {len(wide_attn_keys) - 4} 个")
    else:
        print("  没有旧格式键 (wide_attention_blocks.*.attn.)")

    print("\n" + "="*60)
    print("Narrow Attention Blocks (带 .attn. 的键):")
    if narrow_attn_keys:
        print(f"  旧格式键数: {len(narrow_attn_keys)}")
        for k in narrow_attn_keys[:4]:
            print(f"    - {k}")
        if len(narrow_attn_keys) > 4:
            print(f"    ... 还有 {len(narrow_attn_keys) - 4} 个")
    else:
        print("  没有旧格式键 (narrow_attention_blocks.*.attn.)")

    print("\n" + "="*60)
    # 总结
    has_old_format = bool(compression_attention_keys or pooling_attention_keys or wide_attn_keys or narrow_attn_keys)
    has_new_format = bool(compression_q_proj_keys or pooling_q_proj_keys)

    if has_old_format and not has_new_format:
        print("结论: 文件完全是旧格式，需要转换")
    elif has_old_format and has_new_format:
        print("结论: 文件是混合格式，部分已转换")
    elif not has_old_format and has_new_format:
        print("结论: 文件已经是新格式")
    else:
        print("结论: 无法确定文件格式")


def main():
    if len(sys.argv) < 2:
        print("用法: python diagnose_adapter.py <adapter_file.safetensors>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在: {file_path}")
        sys.exit(1)

    analyze_file(file_path)


if __name__ == "__main__":
    main()
