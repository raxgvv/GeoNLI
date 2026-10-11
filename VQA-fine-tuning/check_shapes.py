
from transformers import AutoProcessor
from PIL import Image
import torch

model_id = "Qwen/Qwen3-VL-8B-Instruct" # Or Qwen2-VL
try:
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
except:
    # Fallback to Qwen2-VL if Qwen3 not available or similar
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True)

image = Image.new('RGB', (224, 224), color='red')
text = "Describe this image."

inputs = processor(text=[text], images=[image], return_tensors="pt", padding=False)

print("Keys:", inputs.keys())
for k, v in inputs.items():
    if isinstance(v, torch.Tensor):
        print(f"{k}: {v.shape}")
    else:
        print(f"{k}: {type(v)}")
