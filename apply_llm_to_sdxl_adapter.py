import torch
import gc
import logging

logger = logging.getLogger("LLM-SDXL-Adapter-Turbo")


class ApplyLLMToSDXLAdapter:
    """
    ComfyUI node that applies loaded LLM to SDXL adapter transformation
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_hidden_states": ("LLM_HIDDEN_STATES",),
                "llm_adapter": ("LLM_ADAPTER",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "info")
    FUNCTION = "apply_adapter"
    CATEGORY = "llm_sdxl"

    def apply_adapter(self, llm_hidden_states, llm_adapter):
        """Apply the LLM to SDXL adapter transformation"""
        try:
            # Get adapter device and dtype
            adapter_device = next(llm_adapter.parameters()).device
            adapter_dtype = next(llm_adapter.parameters()).dtype

            # Move input to adapter device and dtype
            input_tensor = llm_hidden_states.to(device=adapter_device, dtype=adapter_dtype)

            # Apply adapter
            with torch.no_grad():
                conditioning_gpu, pooled_gpu = llm_adapter(input_tensor)

            # Immediately move to CPU and ensure contiguous memory
            conditioning = conditioning_gpu.cpu().contiguous()
            pooled_output = pooled_gpu.cpu().contiguous()

            # Clean up GPU tensors immediately to prevent memory accumulation
            del input_tensor, conditioning_gpu, pooled_gpu
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Format conditioning for ComfyUI
            # ComfyUI expects conditioning as a list of [cond_tensor, metadata_dict] tuples
            comfy_conditioning = [[conditioning, {"pooled_output": pooled_output}]]

            # Prepare info
            info = f"Conditioning shape: {conditioning.shape}"

            logger.info(f"Applied LLM to SDXL adapter: {info}")

            return (comfy_conditioning, info)

        except Exception as e:
            logger.error(f"Failed to apply adapter: {str(e)}")
            raise Exception(f"Adapter application failed: {str(e)}")


# Node mapping for ComfyUI registration
NODE_CLASS_MAPPINGS = {"ApplyLLMToSDXLAdapter": ApplyLLMToSDXLAdapter}

NODE_DISPLAY_NAME_MAPPINGS = {"ApplyLLMToSDXLAdapter": "Apply LLM To SDXL Adapter"}
