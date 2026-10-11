import argparse
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
import os
import sys
import re

TEMPLATES = {
    "binary": {"system": "You are a visual question answering assistant. Answer concisely.", "user_suffix": "Answer with yes or no."},
    "numeric": {"system": "You are a visual question answering assistant. Answer concisely.", "user_suffix": "Return only a number in digits, no units, rounded to one decimal place."},
    "semantic": {"system": "You are a visual question answering assistant. Answer concisely.", "user_suffix": ""}
}

def get_prompt_text(task, question):
    template = TEMPLATES.get(task, TEMPLATES['semantic'])
    return f"<|im_start|>system\n{template['system']}<|im_end|>\n<|im_start|>user\nPicture 1: <|vision_start|><|image_pad|><|vision_end|>\n{template['user_suffix']} {question}<|im_end|>\n<|im_start|>assistant\n"

def load_model(checkpoint_path, device="cuda"):
    print(f"Loading model from {checkpoint_path}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", trust_remote_code=True)
    return model, processor

def validate_output(text, task_type):
    text = text.strip()
    if task_type == "binary":
        text_lower = text.lower()
        # Check for whole words "yes" or "no"
        if re.search(r'\byes\b', text_lower):
            return "Yes"
        elif re.search(r'\bno\b', text_lower):
            return "No"
        return "No" # Default fallback
    elif task_type == "numeric":
        # Find the first number (integer or float)
        match = re.search(r"[-+]?\d*\.\d+|\d+", text)
        if match:
            return match.group()
    return text

def run_inference(model, processor, image_path, question, task_type="semantic"):
    """
    Run inference on a single image and question.
    
    Args:
        model: Loaded Qwen2VL model
        processor: Loaded processor
        image_path: Path to the image file
        question: The question string
        task_type: 'binary', 'numeric', or 'semantic'
    """
    # Prepare prompt
    prompt_text = get_prompt_text(task_type, question)
    
    # Load image
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
    
    # Process inputs
    inputs = processor(
        text=[prompt_text],
        images=[image],
        padding=True,
        return_tensors="pt"
    )
    
    # Move inputs to device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128
        )
    
    # Decode output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0]
    
    return validate_output(output_text, task_type)

def router(question, image_path, task_type):
    """
    Router function to decide which model to use.
    Currently routes all queries to the fine-tuned model.
    """
    # TODO: Implement routing logic here
    # For now, always return 'fine-tuned'
    return "fine-tuned"

def main():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned VQA model")
    parser.add_argument("--checkpoint", type=str, default="checkpoint-10724", help="Path to model checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--question", type=str, required=True, help="Question to ask")
    parser.add_argument("--task", type=str, default="semantic", choices=["binary", "numeric", "semantic"], help="Task type")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found at {args.checkpoint}")
        return

    selected_model = router(args.question, args.image, args.task)

    if selected_model == "fine-tuned":
        model, processor = load_model(args.checkpoint)
        answer = run_inference(model, processor, args.image, args.question, args.task)
    else:
        print(f"Model '{selected_model}' not implemented.")
        return
    
    print("\n" + "="*30)
    print(f"Question: {args.question}")
    print(f"Task: {args.task}")
    print(f"Answer: {answer}")
    print("="*30 + "\n")

if __name__ == "__main__":
    main()
