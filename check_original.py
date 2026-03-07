"""
检查 adapter 文件的详细信息
包括架构参数 (n_wide_blocks, n_narrow_blocks 等) 和权重键名分析
"""
import sys
from safetensors.torch import load_file
import re


def analyze_adapter_structure(keys):
    """分析适配器架构参数"""
    info = {
        'n_wide_blocks': 0,
        'n_narrow_blocks': 0,
        'has_compression': False,
        'has_pooling': False,
        'attention_format': 'unknown',  # 'in_proj' or 'separate_qkv'
        'total_params': 0,
    }

    # 统计 wide attention blocks
    wide_blocks = set()
    narrow_blocks = set()

    for key in keys:
        # Wide blocks: wide_attention_blocks.{idx}.xxx
        match = re.match(r'wide_attention_blocks\.(\d+)\.', key)
        if match:
            wide_blocks.add(int(match.group(1)))

        # Narrow blocks: narrow_attention_blocks.{idx}.xxx
        match = re.match(r'narrow_attention_blocks\.(\d+)\.', key)
        if match:
            narrow_blocks.add(int(match.group(1)))

        # Check for compression
        if 'compression' in key.lower():
            info['has_compression'] = True

        # Check for pooling
        if 'pooling' in key.lower():
            info['has_pooling'] = True

        # Detect attention format
        if '.attn.in_proj' in key or 'attention.in_proj' in key:
            info['attention_format'] = 'in_proj (MultiheadAttention)'
        elif '.q_proj.' in key and info['attention_format'] == 'unknown':
            info['attention_format'] = 'separate_qkv (Flash Attention compatible)'

    info['n_wide_blocks'] = len(wide_blocks)
    info['n_narrow_blocks'] = len(narrow_blocks)
    info['wide_block_indices'] = sorted(wide_blocks) if wide_blocks else []
    info['narrow_block_indices'] = sorted(narrow_blocks) if narrow_blocks else []

    return info


def estimate_dimensions(keys):
    """估计维度参数"""
    dims = {}

    # Try to find seq_projection dimensions
    for key in keys:
        if 'seq_projection.weight' in key:
            # Shape is typically (sdxl_seq_dim, llm_dim)
            dims['seq_projection'] = key

    # Find other projections
    for key in keys:
        if 'pooled_projection.0.weight' in key:
            dims['pooled_projection'] = key

    return dims


def check_file(file_path):
    try:
        state_dict = load_file(file_path)
    except Exception as e:
        print(f"错误: {e}")
        return

    keys = sorted(state_dict.keys())

    print("=" * 70)
    print(f"文件: {file_path}")
    print("=" * 70)

    # 基础统计
    print(f"\n【基础统计】")
    print(f"  总键数: {len(keys)}")

    # 分析架构
    structure = analyze_adapter_structure(keys)

    print(f"\n【架构参数】")
    print(f"  Wide Blocks 数量 (n_wide_blocks): {structure['n_wide_blocks']}")
    if structure['wide_block_indices']:
        print(f"    索引: {structure['wide_block_indices']}")
    print(f"  Narrow Blocks 数量 (n_narrow_blocks): {structure['n_narrow_blocks']}")
    if structure['narrow_block_indices']:
        print(f"    索引: {structure['narrow_block_indices']}")
    print(f"  包含 Compression: {'是' if structure['has_compression'] else '否'}")
    print(f"  包含 Pooling: {'是' if structure['has_pooling'] else '否'}")
    print(f"  Attention 格式: {structure['attention_format']}")

    # 维度信息
    print(f"\n【维度信息】")
    for key in keys:
        if 'seq_projection.weight' in key:
            tensor = state_dict[key]
            print(f"  seq_projection: {tensor.shape}")
            print(f"    → LLM dim: {tensor.shape[1]}")
            print(f"    → SDXL seq dim: {tensor.shape[0]}")
            break

    for key in keys:
        if 'pooled_projection.3.weight' in key:
            tensor = state_dict[key]
            print(f"  pooled_projection (输出): {tensor.shape}")
            print(f"    → SDXL pooled dim: {tensor.shape[0]}")
            break

    # 检查 compression queries 维度
    for key in keys:
        if 'compression_queries' in key:
            tensor = state_dict[key]
            print(f"  compression_queries: {tensor.shape}")
            print(f"    → target_seq_len: {tensor.shape[1]}")
            break

    # 分类统计
    print(f"\n【权重分类统计】")
    categories = {
        'seq_projection': [],
        'input_position_embeddings': [],
        'output_position_embeddings': [],
        'wide_attention_blocks': [],
        'compression': [],
        'narrow_attention_blocks': [],
        'pooling': [],
        'pooled_projection': [],
    }

    for key in keys:
        for cat in categories.keys():
            if key.startswith(cat):
                categories[cat].append(key)
                break

    for cat, cat_keys in categories.items():
        if cat_keys:
            print(f"\n  {cat}: {len(cat_keys)} 个键")
            # 显示前5个
            for k in cat_keys[:5]:
                tensor = state_dict[k]
                print(f"    - {k}: {tuple(tensor.shape)}")
            if len(cat_keys) > 5:
                print(f"    ... 还有 {len(cat_keys) - 5} 个")

    # 特殊格式检测
    print(f"\n【格式检测】")
    in_proj_keys = [k for k in keys if 'in_proj' in k]
    q_proj_keys = [k for k in keys if '.q_proj.' in k]

    if in_proj_keys:
        print(f"  检测到 in_proj 格式键: {len(in_proj_keys)} 个")
        print(f"    (旧版 MultiheadAttention 格式)")
    if q_proj_keys:
        print(f"  检测到 q_proj/k_proj/v_proj 格式键: {len(q_proj_keys)} 个")
        print(f"    (新版 Flash Attention 兼容格式)")

    # 推荐的预设配置
    print(f"\n【推荐的预设配置】")
    print(f"  根据此适配器文件，推荐的配置为:")
    print(f"  {{")
    print(f'      "llm_dim": <从 seq_projection 推断>,')
    print(f'      "sdxl_seq_dim": 2048,')
    print(f'      "sdxl_pooled_dim": 1280,')
    print(f'      "target_seq_len": <从 compression_queries 推断>,')
    print(f'      "n_wide_blocks": {structure["n_wide_blocks"]},')
    print(f'      "n_narrow_blocks": {structure["n_narrow_blocks"]},')
    print(f'      "num_heads": 16,')
    print(f'      "dropout": 0,')
    print(f"  }}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python check_original.py <adapter_file.safetensors>")
        print("\n示例:")
        print("  python check_original.py model.safetensors")
        sys.exit(1)
    check_file(sys.argv[1])
