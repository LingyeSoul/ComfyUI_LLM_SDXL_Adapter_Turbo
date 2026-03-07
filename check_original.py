"""
检查原始 adapter 文件的键名格式
"""
import sys
from safetensors.torch import load_file

def check_file(file_path):
    try:
        state_dict = load_file(file_path)
    except Exception as e:
        print(f"错误: {e}")
        return

    keys = sorted(state_dict.keys())

    print(f"文件: {file_path}")
    print(f"总键数: {len(keys)}")
    print("\n所有键名:")
    for key in keys:
        print(f"  {key}")

    # 分类统计
    compression_in_proj = [k for k in keys if "compression_attention.in_proj" in k]
    compression_q_proj = [k for k in keys if "compression_attention.q_proj" in k]
    pooling_in_proj = [k for k in keys if "pooling_attention.in_proj" in k]
    pooling_q_proj = [k for k in keys if "pooling_attention.q_proj" in k]

    print("\n" + "="*60)
    print("Compression Attention:")
    print(f"  in_proj 格式: {len(compression_in_proj)}")
    for k in compression_in_proj:
        print(f"    - {k}")
    print(f"  q_proj 格式: {len(compression_q_proj)}")
    for k in compression_q_proj:
        print(f"    - {k}")

    print("\nPooling Attention:")
    print(f"  in_proj 格式: {len(pooling_in_proj)}")
    for k in pooling_in_proj:
        print(f"    - {k}")
    print(f"  q_proj 格式: {len(pooling_q_proj)}")
    for k in pooling_q_proj:
        print(f"    - {k}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python check_original.py <adapter_file.safetensors>")
        sys.exit(1)
    check_file(sys.argv[1])
