
from transformers import AutoProcessor
from PIL import Image
import torch

model_id = "Qwen/Qwen3-VL-8B-Instruct"
try:
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
except:
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True)

image = Image.new('RGB', (224, 224), color='red')

# Test 1: <image>
text1 = "<|im_start|>user\n<image>\nDescribe this.<|im_end|>"
inputs1 = processor(text=[text1], images=[image], return_tensors="pt", padding=False)
print(f"Test 1 (<image>): input_ids shape: {inputs1.input_ids.shape}")

# Test 2: <|image_pad|>
text2 = "<|im_start|>user\n<|image_pad|>\nDescribe this.<|im_end|>"
inputs2 = processor(text=[text2], images=[image], return_tensors="pt", padding=False)
print(f"Test 2 (<|image_pad|>): input_ids shape: {inputs2.input_ids.shape}")

# Test 3: <|vision_start|><|image_pad|><|vision_end|>
text3 = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\nDescribe this.<|im_end|>"
inputs3 = processor(text=[text3], images=[image], return_tensors="pt", padding=False)
print(f"Test 3 (Full tags): input_ids shape: {inputs3.input_ids.shape}")
