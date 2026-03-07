import torch
import logging

logger = logging.getLogger("LLM-SDXL-Adapter-Turbo")


class LLMTextEncoder:
    """
    ComfyUI node that encodes text using a loaded Language Model
    Supports various LLM architectures with chat templates
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LLM_MODEL",),
                "tokenizer": ("LLM_TOKENIZER",),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "masterpiece, best quality, 1girl, anime style"
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are expert in understanding of user prompts for image generations. Create an image according to the prompt from user."
                }),
                "use_chat_template": ("BOOLEAN", {
                    "default": True
                }),
                "add_generation_prompt": ("BOOLEAN", {
                    "default": False
                }),
                "hidden_state_mode": (["auto", "on", "off"], {
                    "default": "auto"
                }),
                "skip_first": ("INT", {
                    "default": 27,
                    "min": 0,
                    "max": 100,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("LLM_HIDDEN_STATES", "STRING")
    RETURN_NAMES = ("hidden_states", "info")
    FUNCTION = "encode_text"
    CATEGORY = "llm_sdxl_turbo"
    
    def encode_text(
        self,
        model,
        tokenizer,
        text,
        system_prompt="You are expert in understanding of user prompts for image generations. Create an image according to the prompt from user.",
        use_chat_template=True,
        add_generation_prompt=False,
        hidden_state_mode="auto",
        skip_first=27,
    ):
        """
        Encode text using Language Model and return hidden states
        """
        try:
            # Get model device
            device = next(model.parameters()).device
            
            if use_chat_template:
                # Build a chat-formatted input for instruction-tuned LLMs.
                messages = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": system_prompt}
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text}
                        ]
                    }
                ]

                inputs = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=add_generation_prompt,
                ).to(device)
            else:
                # Raw prompt mode is often better for tag-style prompts.
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                ).to(device)
            
            if hidden_state_mode == "auto":
                hidden_state_only = bool(getattr(model, "_llm_sdxl_hidden_state_only", True))
            else:
                hidden_state_only = hidden_state_mode == "on"
            hidden_state_only_impl = getattr(model, "_llm_sdxl_hidden_state_only_impl", "compat")

            with torch.inference_mode():
                if hidden_state_only:
                    outputs = model(
                        **inputs,
                        use_cache=False,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                else:
                    outputs = model(
                        **inputs,
                        use_cache=False,
                        output_hidden_states=True,
                        return_dict=True,
                    )

            # Keep adapter compatibility: prioritize hidden_states[-1] first.
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                full_hidden_states = outputs.hidden_states[-1]
                hidden_state_source = "hidden_states[-1]"
            elif isinstance(outputs, dict) and outputs.get("hidden_states") is not None:
                full_hidden_states = outputs["hidden_states"][-1]
                hidden_state_source = "hidden_states[-1]"
            elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                full_hidden_states = outputs.last_hidden_state
                hidden_state_source = "last_hidden_state"
            elif isinstance(outputs, dict) and outputs.get("last_hidden_state") is not None:
                full_hidden_states = outputs["last_hidden_state"]
                hidden_state_source = "last_hidden_state"
            else:
                raise RuntimeError("Model output does not contain hidden states")

            # Extract hidden states and skip prefix tokens if requested.
            # Convert to float32 for consistent processing with the adapter
            total_tokens = full_hidden_states.shape[1]
            effective_skip = min(skip_first, max(0, total_tokens - 1))
            hidden_states = full_hidden_states[:, effective_skip:, :].to(torch.float).contiguous()
            # Prepare info
            info = (
                f"Text: {text[:50]}...\n"
                f"Mode: {'chat_template' if use_chat_template else 'raw_prompt'}\n"
                f"add_generation_prompt: {add_generation_prompt if use_chat_template else 'N/A'}\n"
                f"hidden_state_mode: requested={hidden_state_mode}, effective={'on' if hidden_state_only else 'off'}\n"
                f"hidden_state_only_impl: {hidden_state_only_impl}\n"
                f"hidden_state_source: {hidden_state_source}\n"
                f"Tokens total: {total_tokens}\n"
                f"Skip first: requested={skip_first}, effective={effective_skip}\n"
                f"Tokens after skip: {hidden_states.shape[1]}\n"
                f"Shape: {hidden_states.shape}"
            )
            
            logger.info(
                "Encoded text with shape: %s | hidden_state_mode=%s effective=%s impl=%s source=%s | total_tokens=%s skip=%s",
                hidden_states.shape,
                hidden_state_mode,
                "on" if hidden_state_only else "off",
                hidden_state_only_impl,
                hidden_state_source,
                total_tokens,
                effective_skip,
            )
            
            return (hidden_states, info)
            
        except Exception as e:
            logger.error(f"Failed to encode text: {str(e)}")
            raise Exception(f"Text encoding failed: {str(e)}")


# Node mapping for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "LLMTextEncoder": LLMTextEncoder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMTextEncoder": "LLM Text Encoder"
} 