import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoProcessor

# Shared templates
TEMPLATES = {
    "binary": {"system": "You are a visual question answering assistant. Answer concisely.", "user_suffix": "Answer with yes or no."},
    "numeric": {"system": "You are a visual question answering assistant. Answer concisely.", "user_suffix": "Return only a number in digits, no units, rounded to one decimal place."},
    "semantic": {"system": "You are a visual question answering assistant. Answer concisely.", "user_suffix": ""}
}

def get_prompt_text(task, question):
    template = TEMPLATES.get(task, TEMPLATES['semantic'])
    return f"<|im_start|>system\n{template['system']}<|im_end|>\n<|im_start|>user\nPicture 1: <|vision_start|><|image_pad|><|vision_end|>\n{template['user_suffix']} {question}<|im_end|>\n<|im_start|>assistant\n"

class VRSBenchTokenizedDataset(Dataset):
    def __init__(self, jsonl_path, image_dir, processor, max_length=2048):
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        task = item['task']
        
        # Load Image
        # The preprocess step ensures 'image' is an absolute path to an existing file
        image_path = item['image']
        
        # Fallback check just in case
        if not os.path.exists(image_path):
             # Try relative to image_dir if absolute fails
             image_path = os.path.join(self.image_dir, os.path.basename(item['image']))
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception:
            # Create a dummy black image if missing to prevent crash
            image = Image.new('RGB', (224, 224), color='black')

        # Prepare text
        prompt_text = get_prompt_text(task, item['question'])
        answer_text = f"{item['answer_text']}<|im_end|>"
        
        # Tokenize
        # Qwen-VL processor expects 'text' and 'images'
        inputs = self.processor(
            text=[prompt_text],
            images=[image],
            return_tensors="pt",
            padding=False
        )
        
        prompt_input_ids = inputs.input_ids[0]
        
        # Collect all other keys (vision related)
        extra_inputs = {}
        for k, v in inputs.items():
            if k != 'input_ids' and k != 'attention_mask':
                extra_inputs[k] = v
        
        # Tokenize answer (text only)
        answer_input_ids = self.processor.tokenizer(answer_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        
        # Concatenate
        input_ids = torch.cat([prompt_input_ids, answer_input_ids])
        
        # Create labels
        labels = input_ids.clone()
        labels[:len(prompt_input_ids)] = -100 # Mask prompt
        
        result = {
            "input_ids": input_ids,
            "labels": labels,
            "task": task
        }
        result.update(extra_inputs)
        
        return result

class DataCollatorForVRSBench:
    def __init__(self, processor):
        self.processor = processor
        self.pad_token_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 0

    def __call__(self, batch):
        input_ids = [x['input_ids'] for x in batch]
        labels = [x['labels'] for x in batch]
        tasks = [x['task'] for x in batch]
        
        # Pad input_ids and labels
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        
        batch_out = {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "tasks": tasks
        }
        
        # Handle vision keys dynamically
        vision_keys = [k for k in batch[0].keys() if k not in ['input_ids', 'labels', 'task']]
        
        for k in vision_keys:
            values = [x[k] for x in batch]
            if k == 'pixel_values':
                # For Qwen2-VL/Qwen3-VL, pixel_values are concatenated
                batch_out[k] = torch.cat(values, dim=0)
            elif k == 'image_grid_thw':
                # Concatenate
                batch_out[k] = torch.cat(values, dim=0)
            else:
                # Try stack, fallback to cat
                try:
                    batch_out[k] = torch.stack(values)
                except:
                    batch_out[k] = torch.cat(values, dim=0)
            
        return batch_out
