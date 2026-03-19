import torch
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
                "force_cpu_output": ("BOOLEAN", {"default": False}),
                "enable_diagnostics": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "info")
    FUNCTION = "apply_adapter"
    CATEGORY = "llm_sdxl_turbo"

    def apply_adapter(self, llm_hidden_states, llm_adapter, force_cpu_output=False, enable_diagnostics=False):
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

            # Keep outputs on adapter device by default; optionally force CPU.
            if force_cpu_output:
                conditioning = conditioning_gpu.cpu().contiguous()
                pooled_output = pooled_gpu.cpu().contiguous()
            else:
                conditioning = conditioning_gpu.contiguous()
                pooled_output = pooled_gpu.contiguous()

            # Clean up GPU tensors
            del input_tensor, conditioning_gpu, pooled_gpu

            # Format conditioning for ComfyUI
            # ComfyUI expects conditioning as a list of [cond_tensor, metadata_dict] tuples
            comfy_conditioning = [[conditioning, {"pooled_output": pooled_output}]]

            # Prepare info
            info_lines = [
                f"Conditioning shape: {conditioning.shape}",
                f"Output device: {conditioning.device}",
            ]

            if enable_diagnostics:
                # Optional diagnostics for prompt-compliance debugging.
                # This introduces GPU->CPU sync when tensors are on GPU.
                cond_mean = conditioning.mean().item()
                cond_std = conditioning.std().item()
                cond_norm = conditioning.norm().item()
                pooled_mean = pooled_output.mean().item()
                pooled_std = pooled_output.std().item()
                pooled_norm = pooled_output.norm().item()
                info_lines.append(
                    f"cond(mean/std/norm): {cond_mean:.6f}/{cond_std:.6f}/{cond_norm:.6f}"
                )
                info_lines.append(
                    f"pooled(mean/std/norm): {pooled_mean:.6f}/{pooled_std:.6f}/{pooled_norm:.6f}"
                )
            else:
                info_lines.append("Diagnostics: disabled")

            info = "\n".join(info_lines)

            logger.info(f"Applied LLM to SDXL adapter: {info}")

            return (comfy_conditioning, info)

        except Exception as e:
            logger.error(f"Failed to apply adapter: {str(e)}")
            raise Exception(f"Adapter application failed: {str(e)}")


# Node mapping for ComfyUI registration
NODE_CLASS_MAPPINGS = {"ApplyLLMToSDXLAdapter": ApplyLLMToSDXLAdapter}

NODE_DISPLAY_NAME_MAPPINGS = {"ApplyLLMToSDXLAdapter": "Apply LLM To SDXL Adapter"}
