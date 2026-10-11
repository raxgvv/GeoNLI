import os
import logging
import torch
import json
import random
import numpy as np
from pathlib import Path

def setup_logging(log_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, "train.log"))
        ]
    )
    return logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(model, tokenizer, optimizer, scheduler, output_dir, epoch, step, is_final=False):
    if is_final:
        output_dir = os.path.join(output_dir, "checkpoint-final")
    else:
        output_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save optimizer and scheduler
    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    
    # Save training state
    with open(os.path.join(output_dir, "trainer_state.json"), "w") as f:
        json.dump({"epoch": epoch, "step": step}, f)

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, strict=True):
    # Load model weights (usually handled by from_pretrained if starting fresh, 
    # but for resuming we might need this)
    # Here we assume model is already loaded with weights, we just load opt/sched
    
    opt_path = os.path.join(checkpoint_path, "optimizer.pt")
    sched_path = os.path.join(checkpoint_path, "scheduler.pt")
    
    if os.path.exists(opt_path):
        try:
            optimizer.load_state_dict(torch.load(opt_path))
        except ValueError as e:
            if strict:
                raise e
            else:
                print(f"Warning: Could not load optimizer state: {e}. Starting optimizer fresh.")
    
    if os.path.exists(sched_path):
        try:
            scheduler.load_state_dict(torch.load(sched_path))
        except ValueError as e:
            if strict:
                raise e
            else:
                print(f"Warning: Could not load scheduler state: {e}. Starting scheduler fresh.")
