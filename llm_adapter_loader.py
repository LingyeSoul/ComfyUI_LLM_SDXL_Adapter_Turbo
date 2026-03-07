import torch
from safetensors.torch import load_file
import logging
import gc
import os
import re
from .utils import get_llm_adapters, get_llm_adapter_path
from .llm_to_sdxl_adapter import LLMToSDXLAdapter

logger = logging.getLogger("LLM-SDXL-Adapter-Turbo")


def convert_explicit_adapter_to_mha(state_dict):
    """
    Converts Adapter from the explicit form with separate QKV layers
    to the MultiheadAttention (MHA) format used in LLM_to_SDXL_Adapter.

    This handles:
    - Concatenating q_proj, k_proj, v_proj -> in_proj
    - Renaming o_proj -> out_proj
    """
    converted_dict = {}
    mha_buffers = {}

    qkv_pattern = re.compile(r"(.*)\.(q|k|v)_proj\.(weight|bias)")
    out_proj_pattern = re.compile(r"(.*)\.o_proj\.(weight|bias)")

    keys_to_remove = []
    for key, value in state_dict.items():
        qkv_match = qkv_pattern.match(key)
        out_match = out_proj_pattern.match(key)
        if qkv_match:
            base_path, proj_type, param_type = qkv_match.groups()

            if base_path not in mha_buffers:
                mha_buffers[base_path] = {'weight': {}, 'bias': {}}

            mha_buffers[base_path][param_type][proj_type] = value

        elif out_match:
            base_path, param_type = out_match.groups()
            new_key = f"{base_path}.out_proj.{param_type}"
            converted_dict[new_key] = value

        else:
            converted_dict[key] = value

    count_converted = 0
    for base_path, params in mha_buffers.items():
        if all(k in params['weight'] for k in ['q', 'k', 'v']):
            combined_weight = torch.cat([
                params['weight']['q'],
                params['weight']['k'],
                params['weight']['v']
            ], dim=0)
            converted_dict[f"{base_path}.in_proj_weight"] = combined_weight
            count_converted += 1

        if all(k in params['bias'] for k in ['q', 'k', 'v']):
            combined_bias = torch.cat([
                params['bias']['q'],
                params['bias']['k'],
                params['bias']['v']
            ], dim=0)
            converted_dict[f"{base_path}.in_proj_bias"] = combined_bias

    logger.info(f"Converted {count_converted} attn blocks from explicit to MultiheadAttention.")
    return converted_dict
ADAPTER_PRESETS = {
    "gemma": {
        "llm_dim": 1152,
        "sdxl_seq_dim": 2048,
        "sdxl_pooled_dim": 1280,
        "target_seq_len": 308,
        "n_wide_blocks": 2,
        "n_narrow_blocks": 3,
        "num_heads": 16,
        "dropout": 0,
    },
    "t5gemma": {
        "llm_dim": 2304,
        "sdxl_seq_dim": 2048,
        "sdxl_pooled_dim": 1280,
        "target_seq_len": 308,
        "n_wide_blocks": 3,
        "n_narrow_blocks": 3,
        "num_heads": 16,
        "dropout": 0.0,
    },
}


class LLMAdapterLoader:
    """
    ComfyUI node that loads LLM to SDXL adapter
    """

    def __init__(self):
        self.adapter = None
        self.current_adapter_path = None
        self.current_adapter_type = None
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        elif torch.xpu.is_available():
            self.device = 'xpu:0'
        else:
            self.device = 'cpu'

    @classmethod
    def INPUT_TYPES(cls):
        adapters = get_llm_adapters()
        adapter_types = ["gemma", "t5gemma"]
        device_types = ["auto", "cuda:0", "cuda:1", "cpu", "xpu:0", "xpu:1"] if torch.xpu.is_available() else ["auto", "cuda:0", "cuda:1", "cpu"]

        return {
            "required": {
                "adapter_name": (
                    adapters,
                    {"default": adapters[0] if adapters else None},
                ),
                "type": (adapter_types, {"default": "gemma"}),
            },
            "optional": {
                "device": (device_types, {"default": "auto"}),
                "force_reload": ("BOOLEAN", {"default": False}),
                "explicit_attention": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("LLM_ADAPTER", "STRING")
    RETURN_NAMES = ("llm_adapter", "info")
    FUNCTION = "load_adapter"
    CATEGORY = "llm_sdxl_turbo"

    def _clear_forward_hooks(self, module):
        """Clear all forward hooks from a module and its children"""
        # Clear hooks on the module itself
        for hook_attr in ['_forward_hooks', '_forward_pre_hooks', '_backward_hooks']:
            if hasattr(module, hook_attr):
                hook_dict = getattr(module, hook_attr)
                if isinstance(hook_dict, dict):
                    for hook_id in list(hook_dict.keys()):
                        hook_dict.pop(hook_id, None)

        # Recursively clear hooks on all child modules
        for child in module.children():
            self._clear_forward_hooks(child)

    def _cleanup_adapter(self):
        """Thoroughly cleanup adapter resources to prevent memory leaks"""
        if self.adapter is not None:
            # Move to CPU first to release GPU memory
            try:
                if hasattr(self.adapter, "device") and "cuda" in str(
                    self.adapter.device
                ):
                    self.adapter = self.adapter.to("cpu")
            except Exception:
                pass

            # Clear forward hooks if any (handle different nn.Module hook storage patterns)
            self._clear_forward_hooks(self.adapter)

            del self.adapter
            self.adapter = None

        # Double gc.collect() to handle circular references
        gc.collect()
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def load_adapter(self, adapter_name, type, device="auto", force_reload=False, explicit_attention=False):
        """Load and initialize the LLM to SDXL adapter"""
        if device == "auto":
            device = self.device

        adapter_path = get_llm_adapter_path(adapter_name)

        if type not in ADAPTER_PRESETS:
            raise ValueError(f"Unknown adapter type: {type}")
        config = ADAPTER_PRESETS[type]

        try:
            # Check if we need to reload
            if (
                force_reload
                or self.adapter is None
                or self.current_adapter_path != adapter_path
                or self.current_adapter_type != type
            ):
                # Thoroughly clear previous adapter
                self._cleanup_adapter()

                logger.info(f"Loading LLM to SDXL adapter from {adapter_path}")

                # Initialize adapter with specified parameters
                self.adapter = LLMToSDXLAdapter(
                    llm_dim=config["llm_dim"],
                    sdxl_seq_dim=config["sdxl_seq_dim"],
                    sdxl_pooled_dim=config["sdxl_pooled_dim"],
                    target_seq_len=config["target_seq_len"],
                    n_wide_blocks=config["n_wide_blocks"],
                    n_narrow_blocks=config["n_narrow_blocks"],
                    num_heads=config["num_heads"],
                    dropout=config["dropout"],
                )

                # Load checkpoint if file exists
                if os.path.exists(adapter_path):
                    checkpoint = load_file(adapter_path)

                    # Check if checkpoint contains input_norm keys (for backward compatibility)
                    strict_load = not any("input_norm" in k for k in checkpoint.keys())

                    # Load checkpoint - model will auto-convert MHA format if needed
                    self.adapter.load_state_dict(checkpoint, strict=strict_load)
                    logger.info(f"Loaded adapter weights from {adapter_path}")
                else:
                    logger.warning(
                        f"Adapter file not found: {adapter_path}, using initialized weights"
                    )

                # Move to device
                self.adapter.to(device)
                self.adapter.eval()

                self.current_adapter_path = adapter_path
                self.current_adapter_type = type
                logger.info("LLM to SDXL adapter loaded successfully")

            info = (
                f"Adapter: {adapter_path}\n"
                f"Type: {type}\n"
                f"Device: {device}\n"
                f"LLM dim: {config['llm_dim']}\n"
                f"SDXL seq dim: {config['sdxl_seq_dim']}\n"
                f"Target seq len: {config['target_seq_len']}"
            )

            return (self.adapter, info)

        except Exception as e:
            logger.error(f"Failed to load adapter: {str(e)}")
            raise Exception(f"Adapter loading failed: {str(e)}")


# Node mapping for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "LLMAdapterLoader": LLMAdapterLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMAdapterLoader": "LLM Adapter Loader",
}
