#!/bin/bash
# Run evaluation
export PYTHONPATH=.

# Full evaluation
python vrsbench_finetune/eval.py \
    --config configs/prod_2L40S.yaml \
    --checkpoint checkpoints/prod/checkpoint-2681 \
    --out_file predictions.jsonl \
    --save_samples sample_outputs.txt

# For quick testing (uncomment below):
# python vrsbench_finetune/eval.py \
#     --config configs/prod_2L40S.yaml \
#     --checkpoint checkpoints/prod/checkpoint-final \
#     --limit 20 \
#     --save_samples quick_test_samples.txt
