import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
import logging
from .utils import get_llm_checkpoints, get_llm_checkpoint_path

logger = logging.getLogger("LLM-SDXL-Adapter")


def _get_attention_implementation(backend="auto"):
    """
    Get the attention implementation based on backend selection.
    
    Args:
        backend: One of "auto", "eager", "sdpa", "flash_attention_2"
    
    Returns:
        Tuple of (implementation_name, available)
    """
    if backend == "auto":
        if hasattr(torch, 'compile'):
            try:
                return "sdpa", True
            except Exception:
                pass
        return "eager", True
    
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
                "force_reload": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("LLM_MODEL", "LLM_TOKENIZER", "STRING")
    RETURN_NAMES = ("model", "tokenizer", "info")
    FUNCTION = "load_model"
    CATEGORY = "llm_sdxl"

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

    def load_model(self, model_name, device="auto", attention_backend="auto", force_reload=False):
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

                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map=device_map,
                    output_hidden_states=True,
                    trust_remote_code=True,
                    attn_implementation=attn_implementation,
                )

                logger.info(f"Using attention implementation: {attn_implementation}")

                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path, trust_remote_code=True
                )

                self.current_model_path = model_path
                logger.info("Language Model loaded successfully")

            info = f"Model: {model_path}\nDevice: {device}\nAttention: {attn_implementation}\nLoaded: {self.model is not None}"

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
