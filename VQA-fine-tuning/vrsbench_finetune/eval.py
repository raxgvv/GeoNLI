import argparse
import json
import torch
import yaml
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
import math
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BertTokenizer, BertModel
from PIL import Image
def get_prompt_text(task, question):
    # Re-implement here or import from dataset if it was updated correctly
    # Since we updated dataset.py to use <|vision_start|>..., we should ensure eval uses same prompt
    from vrsbench_finetune.dataset import get_prompt_text as get_prompt_text_dataset
    return get_prompt_text_dataset(task, question)

def extract_answer(text):
    # Extract text after <|im_start|>assistant
    if "<|im_start|>assistant" in text:
        return text.split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()
    return text.strip()

class BertBleuScorer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased').to(device)
        self.model.eval()
        self.device = device

    def get_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[0], inputs['attention_mask'][0]

    def compute_score(self, pred, ref):
        pred_emb, pred_mask = self.get_embeddings(pred)
        ref_emb, ref_mask = self.get_embeddings(ref)
        
        pred_tokens = pred_emb[pred_mask == 1]
        ref_tokens = ref_emb[ref_mask == 1]
        
        if len(pred_tokens) > 2: pred_tokens = pred_tokens[1:-1]
        if len(ref_tokens) > 2: ref_tokens = ref_tokens[1:-1]
        
        L_C = len(pred_tokens)
        L_R = len(ref_tokens)
        
        if L_C == 0 or L_R == 0:
            return 0.0
            
        alpha = 0.5
        LP = math.exp(-alpha * abs(L_C - L_R) / L_R)
        
        scores = []
        for n in range(1, 5):
            if L_R < n: 
                scores.append(0.0)
                continue
            if L_C < n:
                scores.append(0.0)
                continue
                
            ref_ngrams = []
            for i in range(L_R - n + 1):
                ngram = ref_tokens[i : i+n].mean(dim=0)
                ref_ngrams.append(ngram)
            ref_ngrams = torch.stack(ref_ngrams)
            
            pred_ngrams = []
            for i in range(L_C - n + 1):
                ngram = pred_tokens[i : i+n].mean(dim=0)
                pred_ngrams.append(ngram)
            pred_ngrams = torch.stack(pred_ngrams)
            
            sim_matrix = F.cosine_similarity(ref_ngrams.unsqueeze(1), pred_ngrams.unsqueeze(0), dim=2)
            max_sims, _ = sim_matrix.max(dim=1)
            P_n = max_sims.mean().item()
            scores.append(P_n)
            
        if not scores:
            return 0.0
            
        S_semantic = LP * max(scores)
        return S_semantic

bert_scorer = None
def get_bert_scorer():
    global bert_scorer
    if bert_scorer is None:
        bert_scorer = BertBleuScorer()
    return bert_scorer

def compute_metrics(predictions, references, tasks):
    metrics = {
        "binary": {"correct": 0, "total": 0, "yes_correct": 0, "yes_total": 0, "no_correct": 0, "no_total": 0},
        "numeric": {"score_sum": 0, "total": 0, "mae": 0, "mse": 0},
        "semantic": {"score_sum": 0, "total": 0}
    }
    
    bert_scorer = None
    
    for pred, ref, task in zip(predictions, references, tasks):
        pred_clean = extract_answer(pred)
        
        if task == "binary":
            metrics["binary"]["total"] += 1
            is_yes_ref = "yes" in ref.lower()
            is_no_ref = "no" in ref.lower()
            
            is_yes_pred = "yes" in pred_clean.lower()
            is_no_pred = "no" in pred_clean.lower()
            
            if is_yes_ref:
                metrics["binary"]["yes_total"] += 1
                if is_yes_pred:
                    metrics["binary"]["yes_correct"] += 1
                    metrics["binary"]["correct"] += 1
            elif is_no_ref:
                metrics["binary"]["no_total"] += 1
                if is_no_pred:
                    metrics["binary"]["no_correct"] += 1
                    metrics["binary"]["correct"] += 1
            else:
                if pred_clean.lower() == ref.lower():
                    metrics["binary"]["correct"] += 1

        elif task == "numeric":
            metrics["numeric"]["total"] += 1
            try:
                p_val = float(re.search(r"-?\d+\.?\d*", pred_clean).group())
                r_val = float(ref)
                
                metrics["numeric"]["mae"] += abs(p_val - r_val)
                metrics["numeric"]["mse"] += (p_val - r_val) ** 2
                
                if r_val != 0:
                    score = math.exp(-23 * abs(p_val - r_val) / abs(r_val))
                else:
                    score = 1.0 if p_val == 0 else 0.0
                
                metrics["numeric"]["score_sum"] += score
                
            except:
                metrics["numeric"].setdefault("failures", 0)
                metrics["numeric"]["failures"] += 1
                
        elif task == "semantic":
            metrics["semantic"]["total"] += 1
            if bert_scorer is None:
                bert_scorer = get_bert_scorer()
            
            score = bert_scorer.compute_score(pred_clean, ref)
            metrics["semantic"]["score_sum"] += score
            
    results = {}
    
    if metrics["binary"]["total"] > 0:
        results["binary_acc"] = metrics["binary"]["correct"] / metrics["binary"]["total"]
    if metrics["binary"]["yes_total"] > 0:
        results["binary_yes_acc"] = metrics["binary"]["yes_correct"] / metrics["binary"]["yes_total"]
    if metrics["binary"]["no_total"] > 0:
        results["binary_no_acc"] = metrics["binary"]["no_correct"] / metrics["binary"]["no_total"]
        
    if metrics["numeric"]["total"] > 0:
        valid_numeric = metrics["numeric"]["total"] - metrics["numeric"].get("failures", 0)
        if valid_numeric > 0:
            results["numeric_mae"] = metrics["numeric"]["mae"] / valid_numeric
            results["numeric_rmse"] = (metrics["numeric"]["mse"] / valid_numeric) ** 0.5
            results["numeric_score"] = metrics["numeric"]["score_sum"] / valid_numeric
        results["numeric_failures"] = metrics["numeric"].get("failures", 0)
        results["numeric_total"] = metrics["numeric"]["total"]
        
    if metrics["semantic"]["total"] > 0:
        results["semantic_bert_bleu"] = metrics["semantic"]["score_sum"] / metrics["semantic"]["total"]
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_file", type=str, default="predictions.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for inference")
    parser.add_argument("--save_samples", type=str, default="sample_outputs.txt", help="File to save sample outputs")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    processor = AutoProcessor.from_pretrained(config['model']['name_or_path'], trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.checkpoint, 
        device_map="auto", 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.eval()
    
    with open(config['data']['val_path'], 'r') as f:
        data = [json.loads(line) for line in f]
        
    if args.limit:
        data = data[:args.limit]
        
    results = []
    refs = []
    tasks = []
    preds = []
    
    for item in tqdm(data):
        task = item['task']
        # Use the imported function which now has the correct <|vision_start|> tags
        prompt_text = get_prompt_text(task, item['question'])
        
        # Load image
        image_path = item['image']
        if not os.path.exists(image_path):
             image_path = os.path.join(config['data']['image_dir'], os.path.basename(item['image']))
        
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), color='black')
        
        # Generate
        inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
        
        # Fix for Qwen3-VL processor output keys
        # Remove batch dimension for vision inputs if processor added it incorrectly for single sample
        # But here we are doing batch size 1 inference, so standard processor output should be fine
        # EXCEPT for the issue we found earlier with dataset.py where we had to remove [0]
        # Let's check if we need to adjust inputs here.
        # The processor returns 'image_grid_thw' as (1, 3) and 'pixel_values' as (patches, dim) usually.
        # But wait, in dataset.py we were manually constructing the batch.
        # Here we use model.generate which expects standard inputs.
        # However, we need to make sure inputs are on device.
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        preds.append(generated_text)
        refs.append(item['answer_text'])
        tasks.append(task)
        
        results.append({
            "question_id": item['ques_id'],
            "prediction": generated_text,
            "reference": item['answer_text'],
            "task": task
        })
        
    # Compute metrics
    metrics = compute_metrics(preds, refs, tasks)
    print(json.dumps(metrics, indent=2))
    
    with open(args.out_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
            
    # Save samples
    with open(args.save_samples, 'w') as f:
        for i, r in enumerate(results):
            f.write(f"Question ID: {r['question_id']}\n")
            f.write(f"Task: {r['task']}\n")
            f.write(f"Reference: {r['reference']}\n")
            f.write(f"Prediction: {r['prediction']}\n")
            f.write("-" * 40 + "\n")

if __name__ == "__main__":
    main()
