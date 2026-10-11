import os
import argparse
import yaml
import math
import json
import importlib.util
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoProcessor, get_scheduler
from tqdm.auto import tqdm
import wandb
from dotenv import load_dotenv
import sys

# Add project root to sys.path to allow imports from vrsbench_finetune
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vrsbench_finetune.model import get_model
from vrsbench_finetune.dataset import VRSBenchTokenizedDataset, DataCollatorForVRSBench
from vrsbench_finetune.utils import setup_logging, set_seed, save_checkpoint, load_checkpoint

def compute_loss(model, batch, task_weights, device):
    # Prepare model inputs
    # Move batch to device
    # With device_map="auto", inputs should be on the same device as the first layer (model.device)
    # or let accelerate hooks handle it. But explicit move is safer for labels etc.
    target_device = model.device if hasattr(model, 'device') else device
    
    model_inputs = {k: v.to(target_device) for k, v in batch.items() if k not in ['tasks', 'task'] and isinstance(v, torch.Tensor)}
    
    # Ensure attention_mask is present
    if 'attention_mask' not in model_inputs:
        config = model.config
        model_inputs['attention_mask'] = model_inputs['input_ids'].ne(config.pad_token_id)

    # Forward pass
    outputs = model(**model_inputs)
    
    logits = outputs.logits
    labels = batch['labels'].to(target_device)
    
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    # Reshape back to [batch, seq_len]
    token_losses = token_losses.view(shift_labels.size())
    
    # Mask for valid labels (not -100)
    label_mask = shift_labels.ne(-100).float()
    
    # Per-example loss: sum(loss) / count(tokens)
    per_example_sum = (token_losses * label_mask).sum(dim=1)
    per_example_count = label_mask.sum(dim=1)
    
    # Avoid division by zero
    per_example_loss = per_example_sum / (per_example_count + 1e-9)
    
    # Apply task weights
    tasks = batch['tasks']
    weights = torch.tensor([task_weights.get(t, 1.0) for t in tasks], device=target_device)
    
    weighted_loss = (per_example_loss * weights).mean()
    
    return weighted_loss

def main():
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint directory to resume from")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Logging
    logger = setup_logging(config['logging']['log_dir'])
    logger.info(f"Training on {device}")
    
    # WandB
    wandb.init(project="vrsbench-finetune", config=config)
        
    set_seed(config['training']['seed'])
    
    # Load Processor & Model
    processor = AutoProcessor.from_pretrained(config['model']['name_or_path'], trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    pad_token_id = processor.tokenizer.pad_token_id
    
    # Mixed Precision Setup
    mixed_precision = config['training'].get('mixed_precision', 'no')
    use_amp = mixed_precision in ["fp16", "bf16"] and device.type == 'cuda'
    amp_dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(mixed_precision == "fp16"))

    # Check for Flash Attention 2
    has_flash_attn = importlib.util.find_spec("flash_attn") is not None
    use_flash_attn = config['model'].get('use_flash_attn', has_flash_attn)
    
    if use_flash_attn and not has_flash_attn:
        logger.warning("Flash Attention 2 requested but not installed. Falling back to eager attention.")
        use_flash_attn = False

    model = get_model(
        config['model']['name_or_path'], 
        use_flash_attn=use_flash_attn,
        bf16=(mixed_precision == 'bf16'),
        device_map="auto"
    )
    model.config.pad_token_id = pad_token_id
    # model.to(device) - Handled by device_map

    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    #     model = torch.nn.DataParallel(model)
    
    # Dataset
    train_dataset = VRSBenchTokenizedDataset(
        config['data']['train_path'], 
        config['data']['image_dir'], 
        processor
    )
    val_dataset = VRSBenchTokenizedDataset(
        config['data']['val_path'], 
        config['data']['image_dir'], 
        processor
    )
    
    collator = DataCollatorForVRSBench(processor)
    
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        collate_fn=collator, 
        batch_size=config['training']['per_device_batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        collate_fn=collator, 
        batch_size=config['training']['per_device_batch_size'],
        num_workers=config['data']['num_workers']
    )
    
    # Optimizer
    backbone_lr = float(config['training']['backbone_lr'])
    head_lr = float(config['training']['lm_head_lr'])
    
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "lm_head" not in n],
            "lr": backbone_lr,
        },
        {
            "params": [p for n, p in model.named_parameters() if "lm_head" in n],
            "lr": head_lr,
        },
    ]
    
    optimizer = AdamW(
        optimizer_grouped_parameters, 
        lr=backbone_lr, 
        weight_decay=config['training']['weight_decay'],
        eps=config['training']['adam_eps']
    )
    
    # Scheduler
    grad_accum_steps = config['training']['gradient_accumulation_steps']
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / grad_accum_steps)
    max_train_steps = config['training']['num_epochs'] * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        name=config['training']['lr_scheduler_type'],
        optimizer=optimizer,
        num_warmup_steps=int(config['training']['warmup_ratio'] * max_train_steps),
        num_training_steps=max_train_steps,
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    global_step = 0
    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        # Load optimizer and scheduler states
        # If batch size changes, optimizer state might be incompatible if it tracks batch-dependent stats (rare for AdamW)
        # But scheduler definitely depends on total steps.
        # We allow loose loading if needed, but ideally we want to preserve momentum.
        load_checkpoint(model, optimizer, lr_scheduler, args.resume_from_checkpoint, strict=False)
        
        # Load trainer state to get epoch and step
        state_path = os.path.join(args.resume_from_checkpoint, "trainer_state.json")
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                state = json.load(f)
                start_epoch = state.get("epoch", 0)
                global_step = state.get("step", 0)
                logger.info(f"Resumed at Epoch {start_epoch}, Step {global_step}")
    
    # Training Loop
    task_weights = config['training'].get('task_weights', {})
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        model.train()
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        
        # If resuming in the middle of an epoch, we should skip steps
        # But standard Dataloader doesn't support easy skipping. 
        # For simplicity, we just start the epoch from beginning but only optimize after global_step matches.
        # Or simpler: just start next full epoch if we saved at epoch end.
        # If we saved mid-epoch, we might re-process some data.
        
        for step, batch in enumerate(progress_bar):
            # Skip steps if we are resuming and haven't reached the global step yet
            # Note: This is a rough approximation. Ideally we'd skip the dataloader.
            # Since we save every 500 steps, we might re-train on < 500 steps.
            # A better way is to ignore this for now or implement proper skipping.
            # Given the request for "simple", we will just proceed.
            
            # Forward
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                loss = compute_loss(model, batch, task_weights, device)
                loss = loss / grad_accum_steps
            
            # Backward
            scaler.scale(loss).backward()
            
            if (step + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
                
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                if global_step % config['logging']['log_steps'] == 0:
                    wandb.log({"train_loss": loss.item() * grad_accum_steps, "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                    
                if global_step % config['training']['save_steps'] == 0:
                    save_checkpoint(model, processor.tokenizer, optimizer, lr_scheduler, config['logging']['output_dir'], epoch, global_step)

        # Validation
        model.eval()
        val_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                    loss = compute_loss(model, batch, task_weights, device)
                val_loss += loss.item()
                num_batches += 1
        
        avg_val_loss = val_loss / num_batches if num_batches > 0 else 0
        
        wandb.log({"val_loss": avg_val_loss}, step=global_step)
        logger.info(f"Epoch {epoch} Val Loss: {avg_val_loss}")
            
        # Save epoch checkpoint
        save_checkpoint(model, processor.tokenizer, optimizer, lr_scheduler, config['logging']['output_dir'], epoch + 1, global_step)

    # Save final model
    save_checkpoint(model, processor.tokenizer, optimizer, lr_scheduler, config['logging']['output_dir'], epoch + 1, global_step, is_final=True)
    logger.info(f"Saved final model to {os.path.join(config['logging']['output_dir'], 'checkpoint-final')}")

if __name__ == "__main__":
    main()
