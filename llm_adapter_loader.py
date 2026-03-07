import torch
from safetensors.torch import load_file
import logging
import gc
import os
from .utils import get_llm_adapters, get_llm_adapter_path
from .llm_to_sdxl_adapter import LLMToSDXLAdapter

logger = logging.getLogger("LLM-SDXL-Adapter-Turbo")

ADAPTER_PRESETS = {
    "gemma": {
        "llm_dim": 1152,
        "sdxl_seq_dim": 2048,
        "sdxl_pooled_dim": 1280,
        "target_seq_len": 308,
        "n_wide_blocks": 2,
        "n_narrow_blocks": 3,
        "num_heads": 16,
        "dropout": 0.1,
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

    def load_adapter(self, adapter_name, type, device="auto", force_reload=False):
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

                    # Force converted adapter format only.
                    # Converted checkpoints use explicit q/k/v proj keys.
                    old_format_patterns = [
                        ".attn.in_proj",
                        ".attn.out_proj",
                        "compression_attention.",
                        "pooling_attention.",
                    ]
                    if any(any(p in k for p in old_format_patterns) for k in checkpoint.keys()):
                        raise ValueError(
                            "Detected legacy unconverted adapter keys. "
                            "Please use a converted adapter safetensors file."
                        )

                    # Temporary debugging mode: force strict load to surface key mismatches.
                    strict_load = True

                    # Load checkpoint and fail fast on any missing/unexpected keys.
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
