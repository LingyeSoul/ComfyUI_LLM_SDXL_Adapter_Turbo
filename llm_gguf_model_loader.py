import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
import logging
import os
import folder_paths
from .utils import get_llm_ggufs, get_llm_gguf_path

logger = logging.getLogger("LLM-SDXL-Adapter-Turbo")

GEMMA_LOCAL_TOKENIZER_DIR = "gemma-3-1b-it"


def _get_local_gemma_tokenizer_path():
    """Return local cache path for the hardcoded Gemma tokenizer."""
    return os.path.join(folder_paths.models_dir, "llm", GEMMA_LOCAL_TOKENIZER_DIR)


def _ensure_local_gemma_tokenizer():
    """
    Ensure hardcoded Gemma tokenizer exists under models/llm/gemma-3-1b-it.
    Offline-only: never download automatically.
    """
    local_path = _get_local_gemma_tokenizer_path()
    required_files = [
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
    ]

    if not os.path.isdir(local_path):
        raise FileNotFoundError(
            "Local Gemma tokenizer directory not found. "
            f"Expected: {local_path}. "
            "Offline mode: auto-download is disabled. "
            "Please place tokenizer files manually."
        )

    missing_files = [
        file_name
        for file_name in required_files
        if not os.path.exists(os.path.join(local_path, file_name))
    ]
    if missing_files:
        raise FileNotFoundError(
            "Local Gemma tokenizer files are incomplete. "
            f"Missing: {', '.join(missing_files)}. "
            f"Expected directory: {local_path}. "
            "Offline mode: auto-download is disabled."
        )

    return local_path


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


class LLMGGUFModelLoader:
    """
    ComfyUI node that loads Language Model and tokenizer
    Supports various LLM architectures (Gemma, Llama, Mistral, etc.)
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_model_path = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Process-level cache to avoid repeated GGUF reloading across node executions.
    _shared_model = None
    _shared_tokenizer = None
    _shared_cache_key = None

    @classmethod
    def INPUT_TYPES(cls):
        ggufs = get_llm_ggufs()
        attention_backends = ["auto", "eager", "sdpa", "flash_attention_2"]
        return {
            "required": {
                "model_name": (ggufs, {"default": ggufs[0] if ggufs else None}),
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
            model_path = get_llm_gguf_path(model_name)
            tokenizer_source = "unknown"
            cache_key = (model_path, model_name, str(device_map), attn_implementation, bool(hidden_state_only))

            if (
                not force_reload
                and LLMGGUFModelLoader._shared_cache_key == cache_key
                and LLMGGUFModelLoader._shared_model is not None
                and LLMGGUFModelLoader._shared_tokenizer is not None
            ):
                self.model = LLMGGUFModelLoader._shared_model
                self.tokenizer = LLMGGUFModelLoader._shared_tokenizer
                self.current_model_path = model_path
                tokenizer_source = _get_local_gemma_tokenizer_path()
                logger.info("Reusing cached GGUF model and tokenizer")
            else:

                # Check if we need to reload
                if (
                    force_reload
                    or self.model is None
                    or self.current_model_path != model_path
                ):
                # Thoroughly clear previous model
                    self._cleanup_model()

                    logger.info(f"Loading Language Model from {model_path}")

                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        gguf_file=model_name,
                        torch_dtype=torch.bfloat16,
                        device_map=device_map,
                        output_hidden_states=True,
                        trust_remote_code=True,
                        attn_implementation=attn_implementation,
                        low_cpu_mem_usage=True,
                    )

                    logger.info(f"Using attention implementation: {attn_implementation}")
                    logger.info(f"Hidden-state-only load path: {hidden_state_only}")

                # Force tokenizer/config source to Gemma IT baseline for better
                # compatibility with adapter expectations.
                    tokenizer_path = _ensure_local_gemma_tokenizer()
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        tokenizer_path,
                        trust_remote_code=True,
                        local_files_only=True,
                    )
                    tokenizer_source = tokenizer_path
                    logger.info(f"Using local tokenizer at: {tokenizer_path}")

                    # Keep behavior consistent with normal loader for text encoder auto mode.
                    self.model.eval()
                    self.model.requires_grad_(False)
                    self.model._llm_sdxl_hidden_state_only = bool(hidden_state_only)
                    self.model._llm_sdxl_hidden_state_only_impl = "compat"

                    self.current_model_path = model_path
                    logger.info("Language Model loaded successfully")
                else:
                    tokenizer_source = _get_local_gemma_tokenizer_path()

                # Update shared cache after successful load/attach.
                if self.model is not None and self.tokenizer is not None:
                    LLMGGUFModelLoader._shared_model = self.model
                    LLMGGUFModelLoader._shared_tokenizer = self.tokenizer
                    LLMGGUFModelLoader._shared_cache_key = cache_key

            model_type = getattr(self.model.config, "model_type", "unknown")
            hidden_size = getattr(self.model.config, "hidden_size", "unknown")
            num_hidden_layers = getattr(self.model.config, "num_hidden_layers", "unknown")

            logger.warning(
                "GGUF model path is active. Quantized/dequantized GGUF hidden states may drift from the adapter training distribution. "
                "For best prompt compliance, prefer LLMModelLoader with the original HF checkpoint."
            )

            info = (
                f"Model: {model_path}\n"
                f"Device: {device}\n"
                f"Attention: {attn_implementation}\n"
                f"hidden_state_only: {hidden_state_only}\n"
                f"hidden_state_only_impl: compat\n"
                f"model_type: {model_type}\n"
                f"hidden_size: {hidden_size}\n"
                f"num_hidden_layers: {num_hidden_layers}\n"
                f"Tokenizer source: {tokenizer_source}\n"
                f"Source: GGUF\n"
                f"Loaded: {self.model is not None}"
            )

            logger.info(f"Tokenizer source: {tokenizer_source}")

            return (self.model, self.tokenizer, info)

        except Exception as e:
            logger.error(f"Failed to load Language Model: {str(e)}")
            raise Exception(f"Model loading failed: {str(e)}")


# Node mapping for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "LLMGGUFModelLoader": LLMGGUFModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMGGUFModelLoader": "LLM GGUF Model Loader",
}
