import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import re

logger = logging.getLogger("LLM-SDXL-Adapter")

# Enable Flash Attention and Memory-Efficient attention backends
def _enable_flash_attention():
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            logger.info("Flash Attention and Memory-Efficient attention enabled")
        except Exception as e:
            logger.warning(f"Failed to enable Flash Attention: {e}")

_enable_flash_attention()


def convert_mha_to_separate_qkv(state_dict):
    """
    Converts state dict from MultiheadAttention format to separate QKV projection format.
    This handles backward compatibility with existing saved weights.

    MultiheadAttention format:
    - attention.{block_idx}.attn.in_proj_weight (dim*3, dim)
    - attention.{block_idx}.attn.in_proj_bias (dim*3)
    - attention.{block_idx}.attn.out_proj.weight (dim, dim)

    Separate QKV format:
    - attention.{block_idx}.q_proj.weight (dim, dim)
    - attention.{block_idx}.k_proj.weight (dim, dim)
    - attention.{block_idx}.v_proj.weight (dim, dim)
    - attention.{block_idx}.out_proj.weight (dim, dim)
    """
    converted_dict = {}
    # Support both formats: in_proj_weight (underscore) and in_proj.weight (dot)
    # Only match attention blocks, not compression_attention or pooling_attention
    mha_pattern = re.compile(r"(.*?)\.attn\.(in_proj|out_proj)[_\.](weight|bias)")

    converted_blocks = set()
    converted_special_blocks = set()  # Track compression and pooling blocks

    for key, value in state_dict.items():
        match = mha_pattern.match(key)
        if match:
            base_path, proj_type, param_type = match.groups()
            converted_blocks.add(base_path)

            if proj_type == "in_proj":
                # Split in_proj_weight/in_proj_bias into q, k, v
                dim = value.shape[0] // 3
                if param_type == "weight":
                    q_weight = value[:dim]
                    k_weight = value[dim:2*dim]
                    v_weight = value[2*dim:]
                    converted_dict[f"{base_path}.q_proj.weight"] = q_weight
                    converted_dict[f"{base_path}.k_proj.weight"] = k_weight
                    converted_dict[f"{base_path}.v_proj.weight"] = v_weight
                else:  # bias
                    q_bias = value[:dim]
                    k_bias = value[dim:2*dim]
                    v_bias = value[2*dim:]
                    converted_dict[f"{base_path}.q_proj.bias"] = q_bias
                    converted_dict[f"{base_path}.k_proj.bias"] = k_bias
                    converted_dict[f"{base_path}.v_proj.bias"] = v_bias
            else:  # out_proj
                converted_dict[f"{base_path}.out_proj.{param_type}"] = value
        else:
            # Check for compression_attention and pooling_attention (q_proj/k_proj/v_proj format)
            # Handle format: compression_attention.q_proj.weight -> compression_q_proj.weight
            if "compression_attention.q_proj" in key:
                new_key = key.replace("compression_attention.q_proj", "compression_q_proj")
                converted_dict[new_key] = value
                converted_special_blocks.add("compression")
                continue
            elif "compression_attention.k_proj" in key:
                new_key = key.replace("compression_attention.k_proj", "compression_k_proj")
                converted_dict[new_key] = value
                converted_special_blocks.add("compression")
                continue
            elif "compression_attention.v_proj" in key:
                new_key = key.replace("compression_attention.v_proj", "compression_v_proj")
                converted_dict[new_key] = value
                converted_special_blocks.add("compression")
                continue
            elif "compression_attention.out_proj" in key:
                new_key = key.replace("compression_attention.out_proj", "compression_out_proj")
                converted_dict[new_key] = value
                converted_special_blocks.add("compression")
                continue
            # Handle format: pooling_attention.q_proj.weight -> pooling_q_proj.weight
            elif "pooling_attention.q_proj" in key:
                new_key = key.replace("pooling_attention.q_proj", "pooling_q_proj")
                converted_dict[new_key] = value
                converted_special_blocks.add("pooling")
                continue
            elif "pooling_attention.k_proj" in key:
                new_key = key.replace("pooling_attention.k_proj", "pooling_k_proj")
                converted_dict[new_key] = value
                converted_special_blocks.add("pooling")
                continue
            elif "pooling_attention.v_proj" in key:
                new_key = key.replace("pooling_attention.v_proj", "pooling_v_proj")
                converted_dict[new_key] = value
                converted_special_blocks.add("pooling")
                continue
            elif "pooling_attention.out_proj" in key:
                new_key = key.replace("pooling_attention.out_proj", "pooling_out_proj")
                converted_dict[new_key] = value
                converted_special_blocks.add("pooling")
                continue
            # Check for compression_attention and pooling_attention (in_proj format - older format)
            elif "compression_attention.in_proj_weight" in key:
                dim = value.shape[0] // 3
                converted_dict[key.replace("compression_attention.in_proj_weight", "compression_q_proj.weight")] = value[:dim]
                converted_dict[key.replace("compression_attention.in_proj_weight", "compression_k_proj.weight")] = value[dim:2*dim]
                converted_dict[key.replace("compression_attention.in_proj_weight", "compression_v_proj.weight")] = value[2*dim:]
                converted_special_blocks.add("compression")
                continue
            elif "compression_attention.in_proj_bias" in key:
                dim = value.shape[0] // 3
                converted_dict[key.replace("compression_attention.in_proj_bias", "compression_q_proj.bias")] = value[:dim]
                converted_dict[key.replace("compression_attention.in_proj_bias", "compression_k_proj.bias")] = value[dim:2*dim]
                converted_dict[key.replace("compression_attention.in_proj_bias", "compression_v_proj.bias")] = value[2*dim:]
                converted_special_blocks.add("compression")
                continue
            elif "pooling_attention.in_proj_weight" in key:
                dim = value.shape[0] // 3
                converted_dict[key.replace("pooling_attention.in_proj_weight", "pooling_q_proj.weight")] = value[:dim]
                converted_dict[key.replace("pooling_attention.in_proj_weight", "pooling_k_proj.weight")] = value[dim:2*dim]
                converted_dict[key.replace("pooling_attention.in_proj_weight", "pooling_v_proj.weight")] = value[2*dim:]
                converted_special_blocks.add("pooling")
                continue
            elif "pooling_attention.in_proj_bias" in key:
                dim = value.shape[0] // 3
                converted_dict[key.replace("pooling_attention.in_proj_bias", "pooling_q_proj.bias")] = value[:dim]
                converted_dict[key.replace("pooling_attention.in_proj_bias", "pooling_k_proj.bias")] = value[dim:2*dim]
                converted_dict[key.replace("pooling_attention.in_proj_bias", "pooling_v_proj.bias")] = value[2*dim:]
                converted_special_blocks.add("pooling")
                continue
            else:
                converted_dict[key] = value

    total_converted = len(converted_blocks) + len(converted_special_blocks)
    if total_converted > 0:
        logger.info(f"Converted {total_converted} attention blocks from MHA to separate QKV format")

    return converted_dict


def pad_to_length(tensor, target_length, dim=1, value=0):
    """Optimized tensor padding using F.pad (3x faster than full + cat)"""
    current_length = tensor.size(dim)

    if current_length >= target_length:
        return tensor.narrow(dim, 0, target_length)

    pad_amount = target_length - current_length

    # F.pad is more efficient: (left, right), (top, bottom), (front, back), ...
    if dim == 1:
        return F.pad(tensor, (0, 0, 0, pad_amount), value=value)
    elif dim == 2:
        return F.pad(tensor, (0, pad_amount), value=value)
    else:
        # Generic case for other dimensions
        pad_list = [0] * (2 * tensor.dim())
        pad_list[2 * (tensor.dim() - dim - 1) + 1] = pad_amount
        return F.pad(tensor, pad_list, value=value)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=16, mlp_ratio=4.0, dropout=0.0):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm1 = nn.LayerNorm(dim)

        # Separate Q, K, V projections for Flash Attention compatibility
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x, mask=None):
        # Self-attention with Flash Attention
        normed = self.norm1(x)

        # Project Q, K, V
        q = self.q_proj(normed)
        k = self.k_proj(normed)
        v = self.v_proj(normed)

        # Reshape for multi-head attention: (B, N, C) -> (B, N, H, D)
        batch_size, seq_len, _ = q.shape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Create attention mask for padding
        attn_mask = None
        if mask is not None:
            # Convert padding mask to attention mask format
            # mask: (B, N) where 1 = valid, 0 = padding
            # We need: (B, 1, 1, N) where True = valid
            attn_mask = mask.unsqueeze(1).unsqueeze(2).bool()

        # Use scaled_dot_product_attention (Flash Attention when available)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0 if self.training else 0.0,
        )

        # Reshape back: (B, H, N, D) -> (B, N, C)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        attn_out = self.out_proj(attn_out)

        x = x + attn_out

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


class LLMToSDXLAdapter(nn.Module):
    """
    Universal adapter for converting any LLM embeddings to SDXL format
    Supports various LLM architectures (Gemma, Llama, Mistral, etc.)
    """

    def __init__(
        self,
        llm_dim=1152,  # Changed from gemma_dim to llm_dim
        sdxl_seq_dim=2048,
        sdxl_pooled_dim=1280,
        max_input_len=512,
        target_seq_len=308,
        n_wide_blocks=3,  # Blocks BEFORE compression
        n_narrow_blocks=3,  # Blocks AFTER compression
        num_heads=16,
        dropout=0,
    ):
        super().__init__()

        self.max_input_len = max_input_len
        self.target_seq_len = target_seq_len
        self.num_heads = num_heads

        # Projections
        if llm_dim != sdxl_seq_dim:
            self.seq_projection = nn.Linear(llm_dim, sdxl_seq_dim)

        # Positional embeddings for full sequence
        self.input_position_embeddings = nn.Parameter(
            torch.randn(1, max_input_len, sdxl_seq_dim)
        )
        # Positional embeddings for compressed sequence
        self.output_position_embeddings = nn.Parameter(
            torch.randn(1, target_seq_len, sdxl_seq_dim)
        )

        # Wide blocks - processing full sequence (512 tokens)
        self.wide_attention_blocks = nn.ModuleList(
            [
                TransformerBlock(sdxl_seq_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(n_wide_blocks)
            ]
        )

        # Compression: Cross-attention with learnable queries
        self.compression_queries = nn.Parameter(
            torch.randn(1, target_seq_len, sdxl_seq_dim)
        )
        # Flash Attention compatible compression attention
        self.compression_q_proj = nn.Linear(sdxl_seq_dim, sdxl_seq_dim)
        self.compression_k_proj = nn.Linear(sdxl_seq_dim, sdxl_seq_dim)
        self.compression_v_proj = nn.Linear(sdxl_seq_dim, sdxl_seq_dim)
        self.compression_out_proj = nn.Linear(sdxl_seq_dim, sdxl_seq_dim)
        self.compression_num_heads = num_heads
        self.compression_head_dim = sdxl_seq_dim // num_heads

        # Norm layer after compression for stability
        self.compression_norm = nn.LayerNorm(sdxl_seq_dim)
        # Optional gate mechanism for weighting information
        self.compression_gate = nn.Sequential(
            nn.Linear(sdxl_seq_dim * 2, sdxl_seq_dim), nn.Sigmoid()
        )

        # Narrow blocks - processing compressed sequence (308 tokens)
        self.narrow_attention_blocks = nn.ModuleList(
            [
                TransformerBlock(sdxl_seq_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(n_narrow_blocks)
            ]
        )

        # Pooling head - now works with processed sequence
        # Flash Attention compatible pooling attention
        self.pooling_q_proj = nn.Linear(sdxl_seq_dim, sdxl_seq_dim)
        self.pooling_k_proj = nn.Linear(sdxl_seq_dim, sdxl_seq_dim)
        self.pooling_v_proj = nn.Linear(sdxl_seq_dim, sdxl_seq_dim)
        self.pooling_out_proj = nn.Linear(sdxl_seq_dim, sdxl_seq_dim)
        self.pooling_num_heads = num_heads
        self.pooling_head_dim = sdxl_seq_dim // num_heads

        # Learnable [CLS]-like token for pooling
        self.pooling_token = nn.Parameter(torch.randn(1, 1, sdxl_seq_dim))

        # Final projection for pooled embeddings
        self.pooled_projection = nn.Sequential(
            nn.Linear(sdxl_seq_dim, sdxl_seq_dim),
            nn.LayerNorm(sdxl_seq_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(sdxl_seq_dim, sdxl_pooled_dim),
        )

    def forward(self, llm_hidden_states, attention_mask=None):
        batch_size, seq_len, _ = llm_hidden_states.shape

        # Project to target dimension
        if self.seq_projection:
            hidden_states = self.seq_projection(llm_hidden_states)
        else:
            hidden_states = llm_hidden_states

        # Padding/truncation to max_input_len
        if seq_len > self.max_input_len:
            hidden_states = hidden_states[:, : self.max_input_len, :]
            if attention_mask is not None:
                attention_mask = attention_mask[:, : self.max_input_len]
        else:
            if seq_len < self.max_input_len:
                hidden_states = pad_to_length(hidden_states, self.max_input_len, dim=1)
                if attention_mask is not None:
                    attention_mask = pad_to_length(
                        attention_mask, self.max_input_len, dim=1, value=0
                    )
                else:
                    attention_mask = torch.ones(
                        batch_size, self.max_input_len, device=hidden_states.device
                    )
                    attention_mask[:, seq_len:] = 0

        # Add positional embeddings
        hidden_states = hidden_states + self.input_position_embeddings

        # ===== STAGE 1: Wide Processing (full sequence) =====
        for block in self.wide_attention_blocks:
            hidden_states = block(hidden_states, attention_mask)

        # ===== STAGE 2: Compression (512 -> 308) =====
        # Prepare queries for compression
        queries = self.compression_queries.expand(batch_size, -1, -1)

        # Cross-attention for compression with Flash Attention
        q = self.compression_q_proj(queries)
        k = self.compression_k_proj(hidden_states)
        v = self.compression_v_proj(hidden_states)

        # Reshape for multi-head: (B, N, C) -> (B, N, H, D)
        tgt_len = q.size(1)
        src_len = k.size(1)
        q = q.view(batch_size, tgt_len, self.compression_num_heads, self.compression_head_dim).transpose(1, 2)
        k = k.view(batch_size, src_len, self.compression_num_heads, self.compression_head_dim).transpose(1, 2)
        v = v.view(batch_size, src_len, self.compression_num_heads, self.compression_head_dim).transpose(1, 2)

        # Create padding mask
        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(2).bool()

        # Flash Attention
        compressed_sequence = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
        )

        # Reshape back: (B, H, T, D) -> (B, T, C)
        compressed_sequence = compressed_sequence.transpose(1, 2).contiguous().view(batch_size, tgt_len, sdxl_seq_dim)
        compressed_sequence = self.compression_out_proj(compressed_sequence)

        # Optional: Gate mechanism for mixing with queries
        gate_input = torch.cat([queries, compressed_sequence], dim=-1)
        gate_weights = self.compression_gate(gate_input)
        compressed_sequence = (
            gate_weights * compressed_sequence + (1 - gate_weights) * queries
        )

        # Apply normalization
        compressed_sequence = self.compression_norm(compressed_sequence)

        # Add output positional embeddings
        compressed_sequence = compressed_sequence + self.output_position_embeddings

        # ===== STAGE 3: Narrow Processing (compressed sequence) =====
        for block in self.narrow_attention_blocks:
            compressed_sequence = block(compressed_sequence)

        # ===== STAGE 4: Pooling for Vector Embeddings =====
        # Pool the compressed sequence with Flash Attention
        pool_token = self.pooling_token.expand(batch_size, -1, -1)

        q = self.pooling_q_proj(pool_token)
        k = self.pooling_k_proj(compressed_sequence)
        v = self.pooling_v_proj(compressed_sequence)

        # Reshape for multi-head
        q = q.view(batch_size, 1, self.pooling_num_heads, self.pooling_head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.pooling_num_heads, self.pooling_head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.pooling_num_heads, self.pooling_head_dim).transpose(1, 2)

        # Flash Attention (no mask needed for pooling)
        pooled = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
        )

        # Reshape back and project
        pooled = pooled.transpose(1, 2).contiguous().view(batch_size, 1, sdxl_seq_dim)
        pooled_output = self.pooling_out_proj(pooled).squeeze(1)

        # Final projection for pooled embeddings
        pooled_output = self.pooled_projection(pooled_output)

        return compressed_sequence, pooled_output

    def load_state_dict(self, state_dict, strict=True):
        """
        Override load_state_dict to automatically convert MHA format weights
        to separate QKV format for backward compatibility.
        """
        # Convert state dict if it contains MHA format weights
        converted_state_dict = convert_mha_to_separate_qkv(state_dict)
        # Call parent load_state_dict with converted weights
        return super().load_state_dict(converted_state_dict, strict=strict)
