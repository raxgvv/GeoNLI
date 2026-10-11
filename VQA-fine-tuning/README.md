# VRSBench Fine-tuning for Qwen3-VL-8B

This repository contains the code to fine-tune Qwen3-VL-8B (or Qwen2-VL) on the VRSBench dataset.

## Setup

1. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

2. Prepare Data:
  ```bash
  python vrsbench_finetune/preprocess.py \
    --input_dir ../VRSBench/Annotations_train \
    --image_dir ../VRSBench/Images_train \
    --out_jsonl data/processed/train.jsonl \
    --numeric_stats data/processed/numeric_stats.json \
    --binary_stats data/processed/binary_stats.json

  python vrsbench_finetune/preprocess.py \
    --input_dir ../VRSBench/Annotations_val \
    --image_dir ../VRSBench/Images_val \
    --out_jsonl data/processed/val.jsonl
  ```

## Training

Run the training script (configured for 2x L40S GPUs):
```bash
bash scripts/run_train.sh
```

Or manually:
```bash
accelerate launch vrsbench_finetune/train.py --config configs/prod_2L40S.yaml
```

## Evaluation

Run the full evaluation:
```bash
bash scripts/run_eval.sh
```

The evaluation script now supports:
- **Metrics**:
  - **Binary**: Accuracy (Overall, Yes-only, No-only)
  - **Numeric**: MAE, RMSE, and Normalized Score ($e^{-23 \cdot \text{rel\_error}}$)
  - **Semantic**: BERT-BLEU Score ($LP \cdot \max(P_n)$)
- **Quick Inference**: Use `--limit N` to run on only N samples.
- **Output Inspection**: Use `--save_samples filename.txt` to save model outputs.

Example for quick check:
```bash
python vrsbench_finetune/eval.py \
  --config configs/prod_2L40S.yaml \
  --checkpoint checkpoints/prod/checkpoint-2681 \
  --limit 10 \
  --save_samples debug_samples.txt
```

## Project Structure

- `vrsbench_finetune/`: Source code
- `preprocess.py`: Data processing
- `dataset.py`: Dataset and Collator
- `model.py`: Model wrapper
- `train.py`: Training loop
- `eval.py`: Evaluation script
- `configs/`: Configuration files
- `scripts/`: Shell scripts
