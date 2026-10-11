import torch
import torch.nn as nn
from transformers import AutoConfig, Qwen3VLForConditionalGeneration

def get_model(model_name_or_path, use_flash_attn=True, bf16=True, device_map=None):
    """
    Loads the Qwen3-VL model.
    """
    # Load model
    # Use low_cpu_mem_usage to avoid OOM during model loading
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if use_flash_attn else "eager",
        low_cpu_mem_usage=True,
        device_map=device_map
    )
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    return model

class QwenVLWrapper(nn.Module):
    """
    Optional wrapper if we need to add custom heads or dropout.
    For full fine-tuning, we usually use the base model directly.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, input_ids, attention_mask=None, pixel_values=None, labels=None, **kwargs):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            **kwargs
        )
