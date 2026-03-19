import torch
import logging

logger = logging.getLogger("LLM-SDXL-Adapter-Turbo")


def _validate_conditioning_item(item, input_name, index):
    """Validate a CONDITIONING entry and return (tensor, meta)."""
    if not isinstance(item, (list, tuple)) or len(item) != 2:
        raise ValueError(
            f"Invalid {input_name}[{index}] format. Expected [tensor, metadata_dict]."
        )

    tensor, meta = item
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"Invalid {input_name}[{index}] tensor type: {type(tensor).__name__}."
        )
    if not isinstance(meta, dict):
        raise TypeError(
            f"Invalid {input_name}[{index}] metadata type: {type(meta).__name__}."
        )

    return tensor, meta


def _iter_paired_conditioning(left, right, left_name, right_name):
    """Yield index-wise pairs using zip semantics with mismatch warnings."""
    left_len = len(left)
    right_len = len(right)

    if left_len != right_len:
        logger.warning(
            "Conditioning length mismatch for %s (%s) vs %s (%s). "
            "Using zip semantics and processing first %s pairs.",
            left_name,
            left_len,
            right_name,
            right_len,
            min(left_len, right_len),
        )

    pair_count = min(left_len, right_len)
    for index in range(pair_count):
        yield (
            index,
            _validate_conditioning_item(left[index], left_name, index),
            _validate_conditioning_item(right[index], right_name, index),
        )


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
                "truncate_strategy": (["disable", "keep_start", "keep_end", "balanced"], {"default": "balanced"}),
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
            token_limit = int(max_tokens)
            combined_conditioning = []
            for index, left_item, right_item in _iter_paired_conditioning(
                conditioning_1,
                conditioning_2,
                "conditioning_1",
                "conditioning_2",
            ):
                cond_1_tensor, cond_1_meta = left_item
                cond_2_tensor, cond_2_meta = right_item

                # Prefer CUDA when available on either input to avoid unnecessary CPU fallback.
                if cond_1_tensor.device.type == "cuda":
                    target_device = cond_1_tensor.device
                elif cond_2_tensor.device.type == "cuda":
                    target_device = cond_2_tensor.device
                else:
                    target_device = cond_1_tensor.device
                target_dtype = cond_1_tensor.dtype

                if cond_1_tensor.device != target_device or cond_1_tensor.dtype != target_dtype:
                    cond_1_tensor = cond_1_tensor.to(device=target_device, dtype=target_dtype)

                if cond_2_tensor.device != target_device:
                    logger.info("Moving conditioning_2[%s] from %s to %s", index, cond_2_tensor.device, target_device)
                    cond_2_tensor = cond_2_tensor.to(device=target_device, dtype=target_dtype)
                elif cond_2_tensor.dtype != target_dtype:
                    cond_2_tensor = cond_2_tensor.to(dtype=target_dtype)

                cond_1_tensor = cond_1_tensor.contiguous()
                cond_2_tensor = cond_2_tensor.contiguous()

                combined_tensor = torch.cat([cond_1_tensor, cond_2_tensor], dim=1)

                combined_tokens = combined_tensor.shape[1]
                if truncate_strategy != "disable" and combined_tokens > token_limit:
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
                        "Token cap applied on pair %s: %s -> %s (%s)",
                        index,
                        combined_tokens,
                        combined_tensor.shape[1],
                        truncate_strategy,
                    )

                combined_meta = cond_1_meta.copy()
                pooled_output = combined_meta.get("pooled_output")
                if isinstance(pooled_output, torch.Tensor):
                    if pooled_output.device != target_device or pooled_output.dtype != target_dtype:
                        combined_meta["pooled_output"] = pooled_output.to(
                            device=target_device,
                            dtype=target_dtype,
                        )

                if "pooled_output" in cond_2_meta and "pooled_output" in cond_1_meta:
                    pooled_1 = cond_1_meta["pooled_output"]
                    pooled_2 = cond_2_meta["pooled_output"]
                    if isinstance(pooled_1, torch.Tensor) and isinstance(pooled_2, torch.Tensor):
                        if pooled_2.shape != pooled_1.shape:
                            logger.warning(
                                "Pooled output shapes differ on pair %s: %s vs %s. Using conditioning_1 pooled_output.",
                                index,
                                pooled_1.shape,
                                pooled_2.shape,
                            )

                for key in ["width", "height", "target_width", "target_height", "crop_w", "crop_h"]:
                    if key in cond_2_meta and key not in combined_meta:
                        combined_meta[key] = cond_2_meta[key]

                combined_conditioning.append([combined_tensor, combined_meta])

                logger.info(
                    "Combined pair %s: %s + %s = %s on %s (max_tokens=%s, strategy=%s)",
                    index,
                    cond_1_tensor.shape,
                    cond_2_tensor.shape,
                    combined_tensor.shape,
                    target_device,
                    token_limit,
                    truncate_strategy,
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
                "truncate_strategy": (["disable", "keep_start", "keep_end", "balanced"], {"default": "balanced"}),
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
            token_limit = int(max_tokens)
            concat_conditioning = []
            for index, left_item, right_item in _iter_paired_conditioning(
                conditioning_to,
                conditioning_from,
                "conditioning_to",
                "conditioning_from",
            ):
                to_tensor, to_meta = left_item
                from_tensor, from_meta = right_item

                if to_tensor.device.type == "cuda":
                    target_device = to_tensor.device
                elif from_tensor.device.type == "cuda":
                    target_device = from_tensor.device
                else:
                    target_device = to_tensor.device
                target_dtype = to_tensor.dtype

                if to_tensor.device != target_device or to_tensor.dtype != target_dtype:
                    to_tensor = to_tensor.to(device=target_device, dtype=target_dtype)

                if from_tensor.device != target_device:
                    logger.info("Moving conditioning_from[%s] from %s to %s", index, from_tensor.device, target_device)
                    from_tensor = from_tensor.to(device=target_device, dtype=target_dtype)
                elif from_tensor.dtype != target_dtype:
                    from_tensor = from_tensor.to(dtype=target_dtype)

                to_tensor = to_tensor.contiguous()
                from_tensor = from_tensor.contiguous()

                concat_tensor = torch.cat([to_tensor, from_tensor], dim=1)

                concat_tokens = concat_tensor.shape[1]
                if truncate_strategy != "disable" and concat_tokens > token_limit:
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
                        "Token cap applied in concat on pair %s: %s -> %s (%s)",
                        index,
                        concat_tokens,
                        concat_tensor.shape[1],
                        truncate_strategy,
                    )

                concat_meta = to_meta.copy()
                pooled_output = concat_meta.get("pooled_output")
                if isinstance(pooled_output, torch.Tensor):
                    if pooled_output.device != target_device or pooled_output.dtype != target_dtype:
                        concat_meta["pooled_output"] = pooled_output.to(
                            device=target_device,
                            dtype=target_dtype,
                        )

                for key in ["width", "height", "target_width", "target_height", "crop_w", "crop_h"]:
                    if key in from_meta and key not in concat_meta:
                        concat_meta[key] = from_meta[key]

                concat_conditioning.append([concat_tensor, concat_meta])

                logger.info(
                    "Concatenated pair %s: %s + %s = %s on %s (max_tokens=%s, strategy=%s)",
                    index,
                    to_tensor.shape,
                    from_tensor.shape,
                    concat_tensor.shape,
                    target_device,
                    token_limit,
                    truncate_strategy,
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
            weight_2 = 1.0 - weight_1
            averaged_conditioning = []
            for index, left_item, right_item in _iter_paired_conditioning(
                conditioning_1,
                conditioning_2,
                "conditioning_1",
                "conditioning_2",
            ):
                tensor_1, meta_1 = left_item
                tensor_2, meta_2 = right_item

                target_device = tensor_1.device
                target_dtype = tensor_1.dtype

                if tensor_2.device != target_device:
                    logger.info("Moving conditioning_2[%s] from %s to %s", index, tensor_2.device, target_device)
                    tensor_2 = tensor_2.to(device=target_device, dtype=target_dtype)
                elif tensor_2.dtype != target_dtype:
                    tensor_2 = tensor_2.to(dtype=target_dtype)

                seq_len_1 = tensor_1.shape[1]
                seq_len_2 = tensor_2.shape[1]

                if seq_len_1 != seq_len_2:
                    if seq_len_1 > seq_len_2:
                        pad_size = seq_len_1 - seq_len_2
                        tensor_2 = torch.nn.functional.pad(tensor_2, (0, 0, 0, pad_size))
                    else:
                        pad_size = seq_len_2 - seq_len_1
                        tensor_1 = torch.nn.functional.pad(tensor_1, (0, 0, 0, pad_size))

                tensor_1 = tensor_1.contiguous()
                tensor_2 = tensor_2.contiguous()
                averaged_tensor = weight_1 * tensor_1 + weight_2 * tensor_2

                averaged_meta = meta_1.copy()

                pooled_1 = averaged_meta.get("pooled_output")
                if isinstance(pooled_1, torch.Tensor):
                    if pooled_1.device != target_device or pooled_1.dtype != target_dtype:
                        pooled_1 = pooled_1.to(device=target_device, dtype=target_dtype)
                        averaged_meta["pooled_output"] = pooled_1

                pooled_2 = meta_2.get("pooled_output")
                if isinstance(pooled_1, torch.Tensor) and isinstance(pooled_2, torch.Tensor):
                    if pooled_2.device != target_device or pooled_2.dtype != target_dtype:
                        pooled_2 = pooled_2.to(device=target_device, dtype=target_dtype)

                    if pooled_1.shape == pooled_2.shape:
                        averaged_meta["pooled_output"] = weight_1 * pooled_1 + weight_2 * pooled_2
                    else:
                        logger.warning(
                            "Pooled output shapes differ on average pair %s: %s vs %s. Using conditioning_1 pooled_output.",
                            index,
                            pooled_1.shape,
                            pooled_2.shape,
                        )

                averaged_conditioning.append([averaged_tensor, averaged_meta])

                logger.info(
                    "Averaged pair %s with weights %.2f/%.2f: shape %s on %s",
                    index,
                    weight_1,
                    weight_2,
                    averaged_tensor.shape,
                    target_device,
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
