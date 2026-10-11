import json
import os
import re
import argparse
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_text(text):
    if not isinstance(text, str):
        return str(text)
    return text.strip()

def is_binary_answer(text):
    text_lower = text.lower()
    if text_lower in ['yes', 'no', 'yes.', 'no.']:
        return True, 1 if 'yes' in text_lower else 0
    return False, None

def parse_numeric_answer(text):
    # Remove commas
    text_clean = text.replace(',', '')
    # Regex for float/int/scientific
    # Matches: -123, 123.45, .45, 1.23e-5
    match = re.search(r'^-?\d*\.?\d+(?:[eE][-+]?\d+)?$', text_clean)
    if match:
        try:
            val = float(match.group())
            return True, val
        except ValueError:
            pass
    
    # Fallback word to number mapping (simple cases)
    word_map = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
    }
    if text_clean.lower() in word_map:
        return True, float(word_map[text_clean.lower()])
        
    return False, None

def process_file(annotation_path, image_dir, split_name):
    with open(annotation_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # VRSBench format: dict with 'qa_pairs' and 'image'
    if isinstance(data, dict):
        qa_list = data.get('qa_pairs', [])
        image_name = data.get('image')
    else:
        return []

    # If image name is missing, fallback to json filename + .png
    if not image_name:
        image_name = f"{Path(annotation_path).stem}.png"

    # Construct absolute image path
    image_path = Path(image_dir) / image_name
    final_image_path = str(image_path.resolve())
    
    processed_qas = []
    
    for qa in qa_list:
        question = qa.get('question', '')
        answer_orig = qa.get('answer', '')
        if isinstance(answer_orig, list):
            answer_orig = answer_orig[0] # Take first if multiple
            
        answer_clean = clean_text(str(answer_orig))
        
        # Task detection
        is_bin, bin_val = is_binary_answer(answer_clean)
        is_num, num_val = parse_numeric_answer(answer_clean)
        
        task = "semantic"
        answer_text = answer_clean
        answer_binary = None
        answer_numeric = None
        
        if is_bin:
            task = "binary"
            answer_binary = bin_val
            answer_text = "yes" if bin_val == 1 else "no"
        elif is_num:
            task = "numeric"
            answer_numeric = num_val
            # Format to 1 decimal place as per prompt spec
            answer_text = f"{num_val:.1f}"
        else:
            task = "semantic"
            # answer_text already set
            
        ques_id = qa.get('ques_id')
        if ques_id is None:
            ques_id = qa.get('question_id', qa.get('id', hash(question)))

        record = {
            "image": final_image_path,
            "question": question,
            "task": task,
            "answer_text": answer_text,
            "answer_binary": answer_binary,
            "answer_numeric": answer_numeric,
            "source_file": str(Path(annotation_path).name),
            "ques_id": ques_id
        }
        processed_qas.append(record)
        
    return processed_qas

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help="Path to Annotations folder")
    parser.add_argument('--image_dir', type=str, required=True, help="Path to Images folder")
    parser.add_argument('--out_jsonl', type=str, required=True, help="Output JSONL file")
    parser.add_argument('--numeric_stats', type=str, help="Output numeric stats JSON")
    parser.add_argument('--binary_stats', type=str, help="Output binary stats JSON")
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    json_files = list(input_path.glob('*.json'))
    
    logger.info(f"Found {len(json_files)} JSON files in {args.input_dir}")
    
    all_records = []
    numeric_values = []
    binary_counts = {"total": 0, "yes": 0, "no": 0}
    
    for jf in tqdm(json_files):
        records = process_file(jf, args.image_dir, input_path.name)
        all_records.extend(records)
        
        for r in records:
            if r['task'] == 'numeric' and r['answer_numeric'] is not None:
                numeric_values.append(r['answer_numeric'])
            elif r['task'] == 'binary':
                binary_counts["total"] += 1
                if r['answer_binary'] == 1:
                    binary_counts["yes"] += 1
                elif r['answer_binary'] == 0:
                    binary_counts["no"] += 1
                
    # Save JSONL
    logger.info(f"Saving {len(all_records)} records to {args.out_jsonl}")
    with open(args.out_jsonl, 'w', encoding='utf-8') as f:
        for r in all_records:
            f.write(json.dumps(r) + '\n')
            
    # Compute stats
    if args.numeric_stats and numeric_values:
        vals = np.array(numeric_values)
        stats = {
            "count": int(len(vals)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "p5": float(np.percentile(vals, 5)),
            "p95": float(np.percentile(vals, 95))
        }
        with open(args.numeric_stats, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Numeric stats saved to {args.numeric_stats}")
        
    # Save binary stats
    if args.binary_stats:
        with open(args.binary_stats, 'w', encoding='utf-8') as f:
            json.dump(binary_counts, f, indent=2)
        logger.info(f"Binary stats saved to {args.binary_stats}")

if __name__ == "__main__":
    main()
