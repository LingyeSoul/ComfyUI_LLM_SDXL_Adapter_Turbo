"""
ComfyUI LLM SDXL Adapter v3.0.1

ComfyUI nodes for using Large Language Models as text encoders for SDXL image generation through trained adapters.
"""

import logging
import sys
import os
from typing import Dict, Tuple

# Setup logging
logger = logging.getLogger("LLM-SDXL-Adapter-Turbo")
logger.setLevel(logging.INFO)

# Add custom formatter with module prefix
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[LLM-SDXL-Adapter-Turbo] %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False  # Prevent duplicate logs from parent loggers

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Check dependencies
try:
    import torch
    import transformers
    import safetensors
    import einops
    logger.info("All required dependencies found")
except ImportError as e:
    logger.error(f"Missing dependency: {e}")
    logger.error("Please install: pip install transformers safetensors einops")
    raise

# Import all node modules from separate files
try:
    from .llm_model_loader import NODE_CLASS_MAPPINGS as LLM_MODEL_LOADER_MAPPINGS
    from .llm_model_loader import NODE_DISPLAY_NAME_MAPPINGS as LLM_MODEL_LOADER_DISPLAY_MAPPINGS
    
    from .llm_gguf_model_loader import NODE_CLASS_MAPPINGS as LLM_GGUF_MODEL_LOADER_MAPPINGS
    from .llm_gguf_model_loader import NODE_DISPLAY_NAME_MAPPINGS as LLM_GGUF_MODEL_LOADER_DISPLAY_MAPPINGS
    
    from .t5gemma_model_loader import NODE_CLASS_MAPPINGS as T5GEMMA_MODEL_LOADER_MAPPINGS
    from .t5gemma_model_loader import NODE_DISPLAY_NAME_MAPPINGS as T5GEMMA_MODEL_LOADER_DISPLAY_MAPPINGS
    
    from .llm_text_encoder import NODE_CLASS_MAPPINGS as LLM_ENCODER_MAPPINGS
    from .llm_text_encoder import NODE_DISPLAY_NAME_MAPPINGS as LLM_ENCODER_DISPLAY_MAPPINGS
    
    from .t5gemma_text_encoder import NODE_CLASS_MAPPINGS as T5GEMMA_ENCODER_MAPPINGS
    from .t5gemma_text_encoder import NODE_DISPLAY_NAME_MAPPINGS as T5GEMMA_ENCODER_DISPLAY_MAPPINGS
    
    from .llm_adapter_loader import NODE_CLASS_MAPPINGS as ADAPTER_LOADER_MAPPINGS
    from .llm_adapter_loader import NODE_DISPLAY_NAME_MAPPINGS as ADAPTER_LOADER_DISPLAY_MAPPINGS
    
    from .llm_adapter_loader_custom import NODE_CLASS_MAPPINGS as ADAPTER_LOADER_CUSTOM_MAPPINGS
    from .llm_adapter_loader_custom import NODE_DISPLAY_NAME_MAPPINGS as ADAPTER_LOADER_CUSTOM_DISPLAY_MAPPINGS
    
    from .apply_llm_to_sdxl_adapter import NODE_CLASS_MAPPINGS as ADAPTER_NODE_MAPPINGS
    from .apply_llm_to_sdxl_adapter import NODE_DISPLAY_NAME_MAPPINGS as ADAPTER_NODE_DISPLAY_MAPPINGS
    
    from .t5gemma_apply_llm_to_sdxl_adapter import NODE_CLASS_MAPPINGS as T5GEMMA_ADAPTER_NODE_MAPPINGS
    from .t5gemma_apply_llm_to_sdxl_adapter import NODE_DISPLAY_NAME_MAPPINGS as T5GEMMA_ADAPTER_NODE_DISPLAY_MAPPINGS
    
    from .conditioning_combine import NODE_CLASS_MAPPINGS as CONDITIONING_COMBINE_MAPPINGS
    from .conditioning_combine import NODE_DISPLAY_NAME_MAPPINGS as CONDITIONING_COMBINE_DISPLAY_MAPPINGS
    
    logger.info("Successfully imported all node modules from separate files")
    
except Exception as e:
    logger.error(f"Failed to import node modules: {e}")
    raise

# Combine all node mappings
NODE_CLASS_MAPPINGS: Dict[str, type] = {}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {}


def _to_turbo_node_id(node_id: str) -> str:
    """Prefix node ids to avoid collisions with the original plugin."""
    return node_id if node_id.startswith("Turbo") else f"Turbo{node_id}"


def _as_bool_env(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


# Optional compatibility mode for existing workflows that still reference old node ids.
# Keep disabled by default to prevent collisions when original + turbo are installed together.
ENABLE_LEGACY_NODE_IDS = _as_bool_env("LLM_SDXL_TURBO_ENABLE_LEGACY_NODE_IDS", "0")


# Add all mappings from separate files
all_mapping_pairs = [
    (LLM_MODEL_LOADER_MAPPINGS, LLM_MODEL_LOADER_DISPLAY_MAPPINGS),
    (LLM_GGUF_MODEL_LOADER_MAPPINGS, LLM_GGUF_MODEL_LOADER_DISPLAY_MAPPINGS),
    (T5GEMMA_MODEL_LOADER_MAPPINGS, T5GEMMA_MODEL_LOADER_DISPLAY_MAPPINGS),
    (LLM_ENCODER_MAPPINGS, LLM_ENCODER_DISPLAY_MAPPINGS),
    (T5GEMMA_ENCODER_MAPPINGS, T5GEMMA_ENCODER_DISPLAY_MAPPINGS),
    (ADAPTER_LOADER_MAPPINGS, ADAPTER_LOADER_DISPLAY_MAPPINGS),
    (ADAPTER_LOADER_CUSTOM_MAPPINGS, ADAPTER_LOADER_CUSTOM_DISPLAY_MAPPINGS),
    (ADAPTER_NODE_MAPPINGS, ADAPTER_NODE_DISPLAY_MAPPINGS),
    (T5GEMMA_ADAPTER_NODE_MAPPINGS, T5GEMMA_ADAPTER_NODE_DISPLAY_MAPPINGS),
    (CONDITIONING_COMBINE_MAPPINGS, CONDITIONING_COMBINE_DISPLAY_MAPPINGS),
]

for class_mapping, display_mapping in all_mapping_pairs:
    for node_id, node_class in class_mapping.items():
        turbo_node_id = _to_turbo_node_id(node_id)
        display_name = display_mapping.get(node_id, node_id)

        if turbo_node_id in NODE_CLASS_MAPPINGS and NODE_CLASS_MAPPINGS[turbo_node_id] is not node_class:
            logger.warning(
                "Turbo node id conflict detected for %s. Existing class will be replaced.",
                turbo_node_id,
            )

        NODE_CLASS_MAPPINGS[turbo_node_id] = node_class
        NODE_DISPLAY_NAME_MAPPINGS[turbo_node_id] = f"{display_name} (Turbo)"

        if ENABLE_LEGACY_NODE_IDS:
            NODE_CLASS_MAPPINGS[node_id] = node_class
            NODE_DISPLAY_NAME_MAPPINGS[node_id] = f"{display_name} (Turbo Legacy ID)"

if ENABLE_LEGACY_NODE_IDS:
    logger.warning(
        "Legacy node ids are enabled via LLM_SDXL_TURBO_ENABLE_LEGACY_NODE_IDS=1. "
        "This may collide with the original plugin if both are installed."
    )

# Version and metadata
__version__ = "3.0.2"
__author__ = "NeuroSenko"
__description__ = "ComfyUI nodes for LLM to SDXL adapter workflow"

# Export information for ComfyUI
WEB_DIRECTORY = "./web"  # For any web UI components (if needed)

# Log successful initialization
logger.info(f"LLM SDXL Adapter v{__version__} initialized successfully")
logger.info(f"Registered {len(NODE_CLASS_MAPPINGS)} nodes from separate files:")
for node_name in sorted(NODE_CLASS_MAPPINGS.keys()):
    display_name = NODE_DISPLAY_NAME_MAPPINGS.get(node_name, node_name)
    logger.info(f"  - {node_name} ({display_name})")

# Custom type definitions for ComfyUI
CUSTOM_TYPES = {
    "LLM_MODEL": "Language Model instance",
    "LLM_TOKENIZER": "Language Model tokenizer instance",
    "LLM_HIDDEN_STATES": "LLM model hidden states",
    "LLM_ADAPTER": "Adapter model instance",
    "LLM_ATTENTION_MASK": "LLM attention mask",
    "VECTOR_CONDITIONING": "SDXL vector conditioning",

}

def get_node_info() -> Dict[str, any]:
    """
    Return information about available nodes for debugging/documentation
    """
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "nodes": {
            name: {
                "display_name": NODE_DISPLAY_NAME_MAPPINGS.get(name, name),
                "class": cls.__name__,
                "category": getattr(cls, "CATEGORY", "unknown") if hasattr(cls, "CATEGORY") else "unknown",
                "function": getattr(cls, "FUNCTION", "unknown") if hasattr(cls, "FUNCTION") else "unknown"
            }
            for name, cls in NODE_CLASS_MAPPINGS.items()
        },
        "custom_types": CUSTOM_TYPES
    }

# Optional: Setup hook for ComfyUI initialization
def setup_js():
    """
    Setup any JavaScript/web components if needed
    """
    pass

# Export what ComfyUI expects
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS", 
    "WEB_DIRECTORY",
    "get_node_info"
]

# Print initialization message
print(f"\n{'='*60}")
print(f"  ComfyUI LLM SDXL Adapter Turbo v{__version__} - Loaded Successfully")
print(f"{'='*60}")
print(f"  Available Nodes: {len(NODE_CLASS_MAPPINGS)}")
print(f"  Main Workflow: LLMModelLoader -> LLMTextEncoder -> LLMAdapterLoader -> ApplyLLMToSDXLAdapter -> KSampler")
print(f"  Supports: Gemma, Llama, Mistral, and other compatible LLMs")
print(f"  Quick Start: Use modular nodes for flexible workflows")
print(f"{'='*60}\n") 