#!/bin/bash
# Run training on single GPU
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python vrsbench_finetune/train.py --config configs/prod_2L40S.yaml

# To resume
# python vrsbench_finetune/train.py --config configs/prod_2L40S.yaml --resume_from_checkpoint checkpoints/prod/checkpoint-2681