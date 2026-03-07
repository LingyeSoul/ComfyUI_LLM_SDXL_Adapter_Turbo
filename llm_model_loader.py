import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
import logging
from .utils import get_llm_checkpoints, get_llm_checkpoint_path

logger = logging.getLogger("LLM-SDXL-Adapter-Turbo")


def _is_flash_attn_2_available():
    """
    检查 Flash Attention 2 是否可用（兼容旧版 transformers）
    """
    try:
        # 方法1: 尝试导入 transformers 的检测函数（新版）
        from transformers.utils import is_flash_attn_2_available
        return is_flash_attn_2_available()
    except ImportError:
        pass
    
    try:
        # 方法2: 尝试导入 transformers 的旧版检测函数
        from transformers.utils import is_flash_attn_available
        return is_flash_attn_available()
    except ImportError:
        pass
    
    try:
        # 方法3: 直接检测 flash_attn 包（兼容旧版 transformers）
        import flash_attn
        return True
    except ImportError:
        return False


def _get_attention_implementation(backend="auto"):
    """
    Get the attention implementation based on backend selection.
    
    Args:
        backend: One of "auto", "eager", "sdpa", "flash_attention_2"
    
    Returns:
        Tuple of (implementation_name, available)
    """
    if backend == "auto":
        # 优先检查 Flash Attention 2
        if _is_flash_attn_2_available():
            logger.info("Flash Attention 2 is available")
            return "flash_attention_2", True
        
        # 其次检查 PyTorch SDPA 是否可用
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            try:
                # 启用 SDPA 后端
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_math_sdp(True)
                
                # 验证是否成功启用
                if torch.backends.cuda.flash_sdp_enabled():
                    logger.info("PyTorch SDPA (Flash Attention) is enabled")
                    return "sdpa", True
            except Exception as e:
                logger.warning(f"Failed to enable SDPA: {e}")
        
        # 最后回退到 eager
        logger.info("Using eager attention implementation")
        return "eager", True
    
    # 用户指定了特定后端
    if backend == "flash_attention_2":
        if not _is_flash_attn_2_available():
            logger.warning("Flash Attention 2 requested but not available. Install with: pip install flash-attn --no-build-isolation")
            return "sdpa", False
    elif backend == "sdpa":
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    return backend, True


class LLMModelLoader:
    """
    ComfyUI node that loads Language Model and tokenizer
    Supports various LLM architectures (Gemma, Llama, Mistral, etc.)
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_model_path = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    @classmethod
    def INPUT_TYPES(cls):
        checkpoints = get_llm_checkpoints()
        attention_backends = ["auto", "eager", "sdpa", "flash_attention_2"]
        return {
            "required": {
                "model_name": (
                    checkpoints,
                    {"default": checkpoints[0] if checkpoints else None},
                ),
            },
            "optional": {
                "device": (["auto", "cuda:0", "cuda:1", "cpu"], {"default": "auto"}),
                "attention_backend": (attention_backends, {"default": "auto"}),
                "hidden_state_only": ("BOOLEAN", {"default": True}),
                "force_reload": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("LLM_MODEL", "LLM_TOKENIZER", "STRING")
    RETURN_NAMES = ("model", "tokenizer", "info")
    FUNCTION = "load_model"
    CATEGORY = "llm_sdxl_turbo"

    def _cleanup_model(self):
        """Thoroughly cleanup model resources to prevent memory leaks"""
        if self.model is not None:
            # Move to CPU first to release GPU memory
            try:
                if hasattr(self.model, "device") and "cuda" in str(self.model.device):
                    self.model = self.model.to("cpu")
            except Exception:
                pass

            # Clear forward hooks if any
            if hasattr(self.model, "_forward_hooks"):
                for hook_id in list(self.model._forward_hooks.keys()):
                    self.model._forward_hooks.pop(hook_id, None)

            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # Double gc.collect() to handle circular references
        gc.collect()
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def load_model(self, model_name, device="auto", attention_backend="auto", hidden_state_only=True, force_reload=False):
        """Load Language Model and tokenizer"""
        if device == "auto":
            device = self.device

        if attention_backend == "auto":
            attention_backend = _get_attention_implementation("auto")[0]

        attn_implementation = attention_backend

        # Convert device to proper device_map format
        if device in ["cuda:0", "cuda:1"]:
            device_map = {"": device}
        elif device == "cpu":
            device_map = "cpu"
        else:
            device_map = device

        try:
            model_path = get_llm_checkpoint_path(model_name)

            # Check if we need to reload
            if (
                force_reload
                or self.model is None
                or self.current_model_path != model_path
            ):
                # Thoroughly clear previous model
                self._cleanup_model()

                logger.info(f"Loading Language Model from {model_path}")

                # Always use compatibility implementation for stable adapter quality.
                model_cls = AutoModelForCausalLM

                model_kwargs = {
                    "torch_dtype": torch.bfloat16,
                    "device_map": device_map,
                    "trust_remote_code": True,
                    "attn_implementation": attn_implementation,
                }
                # Keep compatibility with legacy path that expects hidden_states in outputs.
                model_kwargs["output_hidden_states"] = True

                self.model = model_cls.from_pretrained(model_path, **model_kwargs)

                # Hidden-state-only inference path.
                self.model.eval()
                self.model.requires_grad_(False)
                self.model._llm_sdxl_hidden_state_only = bool(hidden_state_only)
                self.model._llm_sdxl_hidden_state_only_impl = "compat"

                logger.info(f"Using attention implementation: {attn_implementation}")
                logger.info(f"Hidden-state-only load path: {hidden_state_only}")
                logger.info("Hidden-state-only implementation: compat")

                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path, trust_remote_code=True
                )

                self.current_model_path = model_path
                logger.info("Language Model loaded successfully")

            model_type = getattr(self.model.config, "model_type", "unknown")
            hidden_size = getattr(self.model.config, "hidden_size", "unknown")
            num_hidden_layers = getattr(self.model.config, "num_hidden_layers", "unknown")

            info = (
                f"Model: {model_path}\n"
                f"Device: {device}\n"
                f"Attention: {attn_implementation}\n"
                f"hidden_state_only: {hidden_state_only}\n"
                f"hidden_state_only_impl: compat\n"
                f"model_type: {model_type}\n"
                f"hidden_size: {hidden_size}\n"
                f"num_hidden_layers: {num_hidden_layers}\n"
                f"Loaded: {self.model is not None}"
            )

            return (self.model, self.tokenizer, info)

        except Exception as e:
            logger.error(f"Failed to load Language Model: {str(e)}")
            raise Exception(f"Model loading failed: {str(e)}")


# Node mapping for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "LLMModelLoader": LLMModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMModelLoader": "LLM Model Loader",
}
