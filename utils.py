import os
import time
import logging
import folder_paths

logger = logging.getLogger("LLM-SDXL-Adapter-Turbo")

# Cache for file system scans
_CACHE = {}
_CACHE_TTL = 60  # 60 seconds cache TTL
_CACHE_TIMESTAMP = {}


def _get_cached(key, scan_func):
    """Get cached result or scan and cache if not available"""
    current_time = time.time()

    # Check if cache exists and is still valid
    if key in _CACHE:
        if current_time - _CACHE_TIMESTAMP.get(key, 0) < _CACHE_TTL:
            return _CACHE[key]

    # Scan and cache
    result = scan_func()
    _CACHE[key] = result
    _CACHE_TIMESTAMP[key] = current_time
    return result


def _invalidate_cache(key=None):
    """Invalidate cache for specific key or all keys"""
    global _CACHE, _CACHE_TIMESTAMP
    if key is None:
        # Use clear() instead of reassignment to avoid linter warning
        _CACHE.clear()
        _CACHE_TIMESTAMP.clear()
    else:
        _CACHE.pop(key, None)
        _CACHE_TIMESTAMP.pop(key, None)


def get_llm_dict():
    """
    Get the dictionary of LLM checkpoints.
    Keys are the names of the LLM checkpoints, values are the paths to the LLM checkpoints.
    """

    def _scan():
        llm_dict = {}
        if "llm" in folder_paths.folder_names_and_paths:
            llm_paths, _ = folder_paths.folder_names_and_paths["llm"]
        elif os.path.exists(os.path.join(folder_paths.models_dir, "llm")):
            llm_paths = [os.path.join(folder_paths.models_dir, "llm")]
        else:
            llm_paths = [os.path.join(folder_paths.models_dir, "LLM")]

        for llm_path in llm_paths:
            if os.path.exists(llm_path):
                for item in os.listdir(llm_path):
                    item_path = os.path.join(llm_path, item)
                    if os.path.isdir(item_path):
                        # Check if it's a valid model directory (contains config.json or similar)
                        if any(
                            f in os.listdir(item_path)
                            for f in [
                                "config.json",
                                "model.safetensors",
                                "pytorch_model.bin",
                            ]
                        ):
                            llm_dict[item] = item_path
                    elif item.endswith((".safetensors", ".bin", ".pt")):
                        llm_dict[item] = item_path

        return llm_dict

    return _get_cached("llm_dict", _scan)


def get_llm_gguf_dict():
    """
    Get the dictionary of GGUF files.
    Keys are the names of the LLM checkpoints, values are the paths to the LLM checkpoints.
    """

    def _scan():
        llm_gguf_dict = {}
        if "llm" in folder_paths.folder_names_and_paths:
            llm_paths, _ = folder_paths.folder_names_and_paths["llm"]
        elif os.path.exists(os.path.join(folder_paths.models_dir, "llm")):
            llm_paths = [os.path.join(folder_paths.models_dir, "llm")]
        else:
            llm_paths = [os.path.join(folder_paths.models_dir, "LLM")]

        for llm_path in llm_paths:
            if os.path.exists(llm_path):
                for item in os.listdir(llm_path):
                    item_path = os.path.join(llm_path, item)
                    if os.path.isfile(item_path):
                        if item_path.lower().endswith(".gguf"):
                            llm_gguf_dict[item] = llm_path

        return llm_gguf_dict

    return _get_cached("llm_gguf_dict", _scan)


def get_adapters_dict():
    """
    Get the dictionary of LLM adapters.
    Keys are the names of the LLM adapters, values are the paths to the LLM adapters.
    """

    def _scan():
        adapters_dict = {}
        if "llm_adapters" in folder_paths.folder_names_and_paths:
            adapters_paths, _ = folder_paths.folder_names_and_paths["llm_adapters"]
        else:
            adapters_paths = [os.path.join(folder_paths.models_dir, "llm_adapters")]

        for adapters_path in adapters_paths:
            if os.path.exists(adapters_path):
                for item in os.listdir(adapters_path):
                    if item.endswith(".safetensors"):
                        adapters_dict[item] = os.path.join(adapters_path, item)

        return adapters_dict

    return _get_cached("adapters_dict", _scan)


def get_llm_checkpoints():
    """
    Get the list of available LLM checkpoints.
    """
    return list(get_llm_dict().keys())


def get_llm_ggufs():
    """
    Get the list of available LLM checkpoints packed in GGUF.
    """
    return list(get_llm_gguf_dict().keys())


def get_llm_adapters():
    """
    Get the list of available LLM adapters.
    """
    return list(get_adapters_dict().keys())


def get_llm_checkpoint_path(model_name):
    """
    Get the path to a LLM checkpoint.
    """
    llm_dict = get_llm_dict()

    if model_name in llm_dict:
        return llm_dict[model_name]
    else:
        raise ValueError(f"Model {model_name} not found")


def get_llm_gguf_path(model_name):
    """
    Get the path to a LLM checkpoint.
    """
    llm_dict = get_llm_gguf_dict()

    if model_name in llm_dict:
        return llm_dict[model_name]
    else:
        raise ValueError(f"Model {model_name} not found")


def get_llm_adapter_path(adapter_name):
    """
    Get the path to an LLM adapter.
    """
    adapters_dict = get_adapters_dict()

    if adapter_name in adapters_dict:
        return adapters_dict[adapter_name]
    else:
        raise ValueError(f"Adapter {adapter_name} not found")


def refresh_all_caches():
    """Force refresh all cached data"""
    _invalidate_cache()
    logger.info("LLM adapter caches refreshed")
