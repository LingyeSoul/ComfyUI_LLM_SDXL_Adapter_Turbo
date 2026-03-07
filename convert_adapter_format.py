"""
转换 adapter.safetensors 文件格式到新的独立 QKV 格式
用法: python convert_adapter_format.py <input_path> [output_path]
"""

import sys
import os
from safetensors.torch import load_file, save_file
from llm_to_sdxl_adapter import convert_mha_to_separate_qkv


def convert_adapter(input_path, output_path=None):
    """转换 adapter 文件到新的格式"""
    if not os.path.exists(input_path):
        print(f"错误: 文件不存在 {input_path}")
        return False

    # 如果没有指定输出路径，默认覆盖原文件或创建新文件
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_converted{ext}"

    print(f"加载: {input_path}")
    state_dict = load_file(input_path)

    print(f"原始键数量: {len(state_dict)}")

    # 检查是否需要转换
    needs_conversion = False
    old_format_keys = []
    for key in state_dict.keys():
        if any(x in key for x in ['.attn.in_proj', '.attn.out_proj', 'compression_attention.', 'pooling_attention.']):
            needs_conversion = True
            old_format_keys.append(key)

    if not needs_conversion:
        print("文件已经是新格式，无需转换")
        return True

    print(f"发现 {len(old_format_keys)} 个需要转换的键")
    for key in old_format_keys[:10]:  # 只显示前10个
        print(f"  - {key}")
    if len(old_format_keys) > 10:
        print(f"  ... 还有 {len(old_format_keys) - 10} 个")

    # 执行转换
    print("\n转换中...")
    converted_state_dict = convert_mha_to_separate_qkv(state_dict)

    print(f"转换后键数量: {len(converted_state_dict)}")

    # 验证转换结果
    remaining_old_keys = [k for k in converted_state_dict.keys()
                          if any(x in k for x in ['compression_attention.', 'pooling_attention.'])]
    if remaining_old_keys:
        print(f"\n警告: 仍有 {len(remaining_old_keys)} 个旧格式键未转换:")
        for key in remaining_old_keys[:5]:
            print(f"  - {key}")
        if len(remaining_old_keys) > 5:
            print(f"  ... 还有 {len(remaining_old_keys) - 5} 个")
    else:
        print("\n✓ 所有旧格式键都已成功转换")

    # 保存转换后的文件
    print(f"保存到: {output_path}")
    save_file(converted_state_dict, output_path)

    print("转换完成!")
    return True


def main():
    if len(sys.argv) < 2:
        print("用法: python convert_adapter_format.py <input_path> [output_path]")
        print("示例:")
        print("  python convert_adapter_format.py model.safetensors")
        print("  python convert_adapter_format.py model.safetensors model_new.safetensors")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    success = convert_adapter(input_path, output_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
