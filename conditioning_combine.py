import torch
import logging

logger = logging.getLogger("LLM-SDXL-Adapter-Turbo")


class LLMConditioningCombine:
    """
    Custom conditioning combine node that ensures device consistency.
    Merges two conditioning inputs by concatenating their embeddings.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning_1": ("CONDITIONING",),
                "conditioning_2": ("CONDITIONING",),
                "max_tokens": ("INT", {"default": 192, "min": 32, "max": 4096, "step": 8}),
                "truncate_strategy": (["keep_start", "keep_end", "balanced"], {"default": "balanced"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "combine"
    CATEGORY = "llm_sdxl_turbo"

    def combine(self, conditioning_1, conditioning_2, max_tokens=192, truncate_strategy="balanced"):
        """
        Combine two conditioning inputs.
        Both conditionings will be moved to the same device before concatenation.
        """
        try:
            if len(conditioning_1) == 0 or len(conditioning_2) == 0:
                # Return the non-empty one if one is empty
                return (conditioning_1 if len(conditioning_2) == 0 else conditioning_2,)

            # Get the first conditioning pair from each input
            # ComfyUI conditioning format: [[tensor, metadata_dict], ...]
            cond_1_tensor, cond_1_meta = conditioning_1[0]
            cond_2_tensor, cond_2_meta = conditioning_2[0]

            # Prefer CUDA when available on either input to avoid unnecessary CPU fallback.
            if cond_1_tensor.device.type == "cuda":
                target_device = cond_1_tensor.device
            elif cond_2_tensor.device.type == "cuda":
                target_device = cond_2_tensor.device
            else:
                target_device = cond_1_tensor.device
            target_dtype = cond_1_tensor.dtype

            # Move conditionings to the selected target device and dtype.
            if cond_1_tensor.device != target_device or cond_1_tensor.dtype != target_dtype:
                cond_1_tensor = cond_1_tensor.to(device=target_device, dtype=target_dtype)

            if cond_2_tensor.device != target_device:
                logger.info(f"Moving conditioning_2 from {cond_2_tensor.device} to {target_device}")
                cond_2_tensor = cond_2_tensor.to(device=target_device, dtype=target_dtype)
            elif cond_2_tensor.dtype != target_dtype:
                cond_2_tensor = cond_2_tensor.to(dtype=target_dtype)

            # Ensure contiguous memory layout
            cond_1_tensor = cond_1_tensor.contiguous()
            cond_2_tensor = cond_2_tensor.contiguous()

            # Concatenate on sequence dimension then enforce an explicit token budget.
            combined_tensor = torch.cat([cond_1_tensor, cond_2_tensor], dim=1)

            token_limit = int(max_tokens)
            combined_tokens = combined_tensor.shape[1]
            if combined_tokens > token_limit:
                if truncate_strategy == "keep_start":
                    combined_tensor = combined_tensor[:, :token_limit, :]
                elif truncate_strategy == "keep_end":
                    combined_tensor = combined_tensor[:, -token_limit:, :]
                else:
                    seq_len_1 = cond_1_tensor.shape[1]
                    seq_len_2 = cond_2_tensor.shape[1]

                    keep_1 = min(seq_len_1, token_limit // 2)
                    keep_2 = min(seq_len_2, token_limit - keep_1)

                    remaining = token_limit - (keep_1 + keep_2)
                    if remaining > 0 and keep_1 < seq_len_1:
                        extra_1 = min(seq_len_1 - keep_1, remaining)
                        keep_1 += extra_1
                        remaining -= extra_1
                    if remaining > 0 and keep_2 < seq_len_2:
                        keep_2 += min(seq_len_2 - keep_2, remaining)

                    combined_tensor = torch.cat(
                        [cond_1_tensor[:, :keep_1, :], cond_2_tensor[:, :keep_2, :]],
                        dim=1,
                    )

                logger.info(
                    "Token cap applied: %s -> %s (%s)",
                    combined_tokens,
                    combined_tensor.shape[1],
                    truncate_strategy,
                )

            # Merge metadata - pooled_output from the first conditioning takes precedence
            combined_meta = cond_1_meta.copy()

            # Keep pooled output colocated with the combined tensor when present.
            pooled_output = combined_meta.get("pooled_output")
            if isinstance(pooled_output, torch.Tensor):
                if pooled_output.device != target_device or pooled_output.dtype != target_dtype:
                    combined_meta["pooled_output"] = pooled_output.to(
                        device=target_device,
                        dtype=target_dtype,
                    )
            
            # If second conditioning has pooled_output, we keep the first one's
            # but log if they're different
            if "pooled_output" in cond_2_meta and "pooled_output" in cond_1_meta:
                if cond_2_meta["pooled_output"].shape != cond_1_meta["pooled_output"].shape:
                    logger.warning(
                        f"Pooled output shapes differ: {cond_1_meta['pooled_output'].shape} vs {cond_2_meta['pooled_output'].shape}. "
                        "Using the first conditioning's pooled_output."
                    )

            # Handle SDXL-specific metadata (width, height, crop, etc.)
            # Use values from first conditioning, but allow override from second if first doesn't have them
            for key in ["width", "height", "target_width", "target_height", "crop_w", "crop_h"]:
                if key in cond_2_meta and key not in combined_meta:
                    combined_meta[key] = cond_2_meta[key]

            # Create combined conditioning
            combined_conditioning = [[combined_tensor, combined_meta]]

            logger.info(
                f"Combined conditioning: {cond_1_tensor.shape} + {cond_2_tensor.shape} = {combined_tensor.shape} "
                f"on device {target_device} (max_tokens={token_limit}, strategy={truncate_strategy})"
            )

            return (combined_conditioning,)

        except Exception as e:
            logger.error(f"Failed to combine conditioning: {str(e)}")
            raise Exception(f"Conditioning combine failed: {str(e)}")


class LLMConditioningConcat:
    """
    Custom conditioning concat node that appends one conditioning to another.
    Similar to combine but with more control over the concatenation axis.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning_to": ("CONDITIONING",),
                "conditioning_from": ("CONDITIONING",),
                "max_tokens": ("INT", {"default": 192, "min": 32, "max": 4096, "step": 8}),
                "truncate_strategy": (["keep_start", "keep_end", "balanced"], {"default": "balanced"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "concat"
    CATEGORY = "llm_sdxl_turbo"

    def concat(self, conditioning_to, conditioning_from, max_tokens=192, truncate_strategy="balanced"):
        """
        Concatenate conditioning_from to conditioning_to.
        This is functionally similar to combine but follows ComfyUI's concat convention.
        """
        try:
            if len(conditioning_to) == 0:
                return (conditioning_from,)
            if len(conditioning_from) == 0:
                return (conditioning_to,)

            # Get tensors and metadata
            to_tensor, to_meta = conditioning_to[0]
            from_tensor, from_meta = conditioning_from[0]

            # Prefer CUDA when either input is on GPU to avoid CPU demotion.
            if to_tensor.device.type == "cuda":
                target_device = to_tensor.device
            elif from_tensor.device.type == "cuda":
                target_device = from_tensor.device
            else:
                target_device = to_tensor.device
            target_dtype = to_tensor.dtype

            # Move both tensors to the selected device and dtype.
            if to_tensor.device != target_device or to_tensor.dtype != target_dtype:
                to_tensor = to_tensor.to(device=target_device, dtype=target_dtype)

            if from_tensor.device != target_device:
                logger.info(f"Moving conditioning_from from {from_tensor.device} to {target_device}")
                from_tensor = from_tensor.to(device=target_device, dtype=target_dtype)
            elif from_tensor.dtype != target_dtype:
                from_tensor = from_tensor.to(dtype=target_dtype)

            # Ensure contiguous memory layout
            to_tensor = to_tensor.contiguous()
            from_tensor = from_tensor.contiguous()

            # Concatenate along sequence dimension
            concat_tensor = torch.cat([to_tensor, from_tensor], dim=1)

            token_limit = int(max_tokens)
            concat_tokens = concat_tensor.shape[1]
            if concat_tokens > token_limit:
                if truncate_strategy == "keep_start":
                    concat_tensor = concat_tensor[:, :token_limit, :]
                elif truncate_strategy == "keep_end":
                    concat_tensor = concat_tensor[:, -token_limit:, :]
                else:
                    seq_len_to = to_tensor.shape[1]
                    seq_len_from = from_tensor.shape[1]

                    keep_to = min(seq_len_to, token_limit // 2)
                    keep_from = min(seq_len_from, token_limit - keep_to)

                    remaining = token_limit - (keep_to + keep_from)
                    if remaining > 0 and keep_to < seq_len_to:
                        extra_to = min(seq_len_to - keep_to, remaining)
                        keep_to += extra_to
                        remaining -= extra_to
                    if remaining > 0 and keep_from < seq_len_from:
                        keep_from += min(seq_len_from - keep_from, remaining)

                    concat_tensor = torch.cat(
                        [to_tensor[:, :keep_to, :], from_tensor[:, :keep_from, :]],
                        dim=1,
                    )

                logger.info(
                    "Token cap applied in concat: %s -> %s (%s)",
                    concat_tokens,
                    concat_tensor.shape[1],
                    truncate_strategy,
                )

            # Use metadata from conditioning_to (the base)
            concat_meta = to_meta.copy()

            # Keep pooled_output colocated with the concatenated tensor when present.
            pooled_output = concat_meta.get("pooled_output")
            if isinstance(pooled_output, torch.Tensor):
                if pooled_output.device != target_device or pooled_output.dtype != target_dtype:
                    concat_meta["pooled_output"] = pooled_output.to(
                        device=target_device,
                        dtype=target_dtype,
                    )

            # Handle SDXL-specific metadata
            for key in ["width", "height", "target_width", "target_height", "crop_w", "crop_h"]:
                if key in from_meta and key not in concat_meta:
                    concat_meta[key] = from_meta[key]

            concat_conditioning = [[concat_tensor, concat_meta]]

            logger.info(
                f"Concatenated conditioning: {to_tensor.shape} + {from_tensor.shape} = {concat_tensor.shape} "
                f"on device {target_device} (max_tokens={token_limit}, strategy={truncate_strategy})"
            )

            return (concat_conditioning,)

        except Exception as e:
            logger.error(f"Failed to concatenate conditioning: {str(e)}")
            raise Exception(f"Conditioning concat failed: {str(e)}")


class LLMConditioningAverage:
    """
    Custom conditioning average node that averages two conditioning inputs.
    Useful for blending prompts.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning_1": ("CONDITIONING",),
                "conditioning_2": ("CONDITIONING",),
                "weight_1": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "average"
    CATEGORY = "llm_sdxl_turbo"

    def average(self, conditioning_1, conditioning_2, weight_1):
        """
        Average two conditioning inputs with specified weight.
        weight_1: weight for conditioning_1 (0.0 to 1.0)
        conditioning_2 weight will be (1 - weight_1)
        """
        try:
            if len(conditioning_1) == 0 or len(conditioning_2) == 0:
                raise ValueError("Both conditionings must be non-empty for averaging")

            # Get tensors and metadata
            tensor_1, meta_1 = conditioning_1[0]
            tensor_2, meta_2 = conditioning_2[0]

            # Determine target device
            target_device = tensor_1.device
            target_dtype = tensor_1.dtype

            # Move tensor_2 to the same device and dtype
            if tensor_2.device != target_device:
                logger.info(f"Moving conditioning_2 from {tensor_2.device} to {target_device}")
                tensor_2 = tensor_2.to(device=target_device, dtype=target_dtype)
            elif tensor_2.dtype != target_dtype:
                tensor_2 = tensor_2.to(dtype=target_dtype)

            # Ensure same sequence length by padding or truncating
            seq_len_1 = tensor_1.shape[1]
            seq_len_2 = tensor_2.shape[1]

            if seq_len_1 != seq_len_2:
                if seq_len_1 > seq_len_2:
                    # Pad tensor_2
                    pad_size = seq_len_1 - seq_len_2
                    tensor_2 = torch.nn.functional.pad(tensor_2, (0, 0, 0, pad_size))
                else:
                    # Pad tensor_1
                    pad_size = seq_len_2 - seq_len_1
                    tensor_1 = torch.nn.functional.pad(tensor_1, (0, 0, 0, pad_size))

            # Ensure contiguous memory layout
            tensor_1 = tensor_1.contiguous()
            tensor_2 = tensor_2.contiguous()

            # Weighted average
            weight_2 = 1.0 - weight_1
            averaged_tensor = weight_1 * tensor_1 + weight_2 * tensor_2

            # Use metadata from conditioning_1
            averaged_meta = meta_1.copy()

            # Average pooled_output if both have it
            if "pooled_output" in meta_1 and "pooled_output" in meta_2:
                pooled_1 = meta_1["pooled_output"]
                pooled_2 = meta_2["pooled_output"]
                
                # Move pooled_2 to same device if needed
                if pooled_2.device != pooled_1.device:
                    pooled_2 = pooled_2.to(device=pooled_1.device, dtype=pooled_1.dtype)
                elif pooled_2.dtype != pooled_1.dtype:
                    pooled_2 = pooled_2.to(dtype=pooled_1.dtype)
                
                averaged_meta["pooled_output"] = weight_1 * pooled_1 + weight_2 * pooled_2

            averaged_conditioning = [[averaged_tensor, averaged_meta]]

            logger.info(
                f"Averaged conditioning with weights {weight_1:.2f}/{weight_2:.2f}: "
                f"shape {averaged_tensor.shape} on device {target_device}"
            )

            return (averaged_conditioning,)

        except Exception as e:
            logger.error(f"Failed to average conditioning: {str(e)}")
            raise Exception(f"Conditioning average failed: {str(e)}")


# Node mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "LLMConditioningCombine": LLMConditioningCombine,
    "LLMConditioningConcat": LLMConditioningConcat,
    "LLMConditioningAverage": LLMConditioningAverage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMConditioningCombine": "LLM Conditioning Combine",
    "LLMConditioningConcat": "LLM Conditioning Concat",
    "LLMConditioningAverage": "LLM Conditioning Average",
}
