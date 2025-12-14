import os
import pickle
import argparse
import json
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.optim as optim
import numpy as np
import random
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

from peft import LoraConfig, get_peft_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


def get_args():
    """Defines and parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune a language model.")

    # Model and Data paths
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-0.6B-Base', help='The name of the pretrained model to use.')
    parser.add_argument('--local_model_dir', type=str, default='./qwen3_local', help='Local directory to prefer for the pretrained model.')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory where the data is stored.')
    parser.add_argument('--output_dir', type=str, default='out-instruction-tuning', help='Directory to save the fine-tuned model.')

    # Training Hyperparameters
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for the optimizer.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for the optimizer.')
    parser.add_argument('--beta1', type=float, default=0.9, help='AdamW optimizer beta1.')
    parser.add_argument('--beta2', type=float, default=0.95, help='AdamW optimizer beta2.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training and validation.')
    parser.add_argument('--grad_accumulation_steps', type=int, default=4, help='Number of steps to accumulate gradients.')

    # Logging and Evaluation
    parser.add_argument('--log_interval', type=int, default=10, help='Log training loss every N steps.')
    parser.add_argument('--eval_interval', type=int, default=50, help='Run validation every N steps.')

    # Optimization method
    parser.add_argument('--optimization_method', type=str, default='adam', choices=['adam', 'sgd', 'lora'], help='Optimization method to use.')

    parser.add_argument('--lora_rank', type=int, default=8, help='The rank of the LoRA matrices.')
    
    # DDP arguments
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    parser.add_argument('--ddp', action='store_true', help='Enable Distributed Data Parallel training')
    
    # Model saving
    parser.add_argument('--save_model', action='store_true', help='Save model checkpoints (default: False, only save logs)')
    parser.add_argument('--save_each_epoch', action='store_true', help='Save model at the end of each epoch (requires --save_model)')

    return parser.parse_args()

class TokenizedDataset(Dataset):
    """A simple dataset class to load tokenized IDs from a pickle file."""
    def __init__(self, pickle_file_path):
        if not os.path.exists(pickle_file_path):
            raise FileNotFoundError(
                f"Pickle file not found at {pickle_file_path}. "
                "Please run the data preparation script first."
            )
        with open(pickle_file_path, 'rb') as f:
            self.data = pickle.load(f)
        print(f"Loaded {len(self.data)} examples from {pickle_file_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class SmartDataCollator:
    """
    Pads sequences to the max length in a batch and creates labels.
    Labels are -100 for pad tokens.
    """
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        input_ids = [torch.tensor(item['input_ids']) for item in batch]
        attention_masks = [torch.tensor(item['attention_mask']) for item in batch]
        labels = [torch.tensor(item['labels']) for item in batch]

        padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        padded_attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            'input_ids': padded_input_ids,
            'attention_mask': padded_attention_masks,
            'labels': padded_labels
        }

def main():
    args = get_args()
    
    # Initialize DDP if enabled
    ddp_enabled = args.ddp or ('RANK' in os.environ and 'WORLD_SIZE' in os.environ)
    if ddp_enabled:
        # Get rank from environment (set by torchrun)
        if 'RANK' in os.environ:
            rank = int(os.environ['RANK'])
            local_rank = int(os.environ['LOCAL_RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
        else:
            # Fallback to local_rank argument
            rank = args.local_rank
            local_rank = args.local_rank
            world_size = torch.cuda.device_count()
        
        # Initialize process group
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        is_main_process = (rank == 0)
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main_process = True

    # Derived paths
    current_file_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(current_file_path)
    data_dir = os.path.join(script_dir, args.data_dir)
    train_data_path = os.path.join(data_dir, 'train.pkl')
    val_data_path = os.path.join(data_dir, 'val.pkl')
    output_dir = os.path.join(script_dir, args.output_dir)

    model_source = args.model_name
    used_local_model = False
    if args.local_model_dir:
        local_candidate = os.path.expanduser(args.local_model_dir)
        if not os.path.isabs(local_candidate):
            local_candidate = os.path.join(script_dir, local_candidate)
        local_candidate = os.path.abspath(local_candidate)

        if os.path.isdir(local_candidate):
            model_source = local_candidate
            used_local_model = True
            if is_main_process:
                print(f"Using local model directory {model_source}")
        else:
            if is_main_process:
                print(f"Local model directory {local_candidate} not found, falling back to {args.model_name}")
    if not used_local_model and is_main_process:
        print(f"Loading model and tokenizer from {model_source}...")
    
    if ddp_enabled and is_main_process:
        print(f"DDP Training: World Size = {world_size}, Rank = {rank}, Local Rank = {local_rank}")
    
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    # device_type for autocast must be a string, not torch.device
    device_type_str = "cuda" if torch.cuda.is_available() else "cpu"

    # Only load tokenizer on main process, but all processes need it
    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        trust_remote_code=True,
        dtype=dtype
    ).to(device)
    
    # Wrap model with DDP if enabled
    if ddp_enabled:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if ddp_enabled:
            model.module.config.pad_token_id = model.module.config.eos_token_id
        else:
            model.config.pad_token_id = model.config.eos_token_id
        if is_main_process:
            print("Set pad_token to eos_token")

    collate_fn = SmartDataCollator(pad_token_id=tokenizer.pad_token_id)

    train_dataset = TokenizedDataset(train_data_path)
    val_dataset = TokenizedDataset(val_data_path)

    # Use DistributedSampler for DDP
    if ddp_enabled:
        train_sampler = DistributedSampler(
            train_dataset, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            sampler=train_sampler,
            shuffle=False  # Shuffle is handled by DistributedSampler
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            sampler=val_sampler
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn
        )

    if is_main_process:
        print(f"Setting up optimizer: {args.optimization_method}")

    # Get model parameters (unwrap DDP if needed)
    model_for_optimizer = model.module if ddp_enabled else model
    
    if args.optimization_method == "adam":
        optimizer = optim.AdamW(
            model_for_optimizer.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2)
        )
    elif args.optimization_method == "sgd":
        optimizer = optim.SGD(
            model_for_optimizer.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            momentum=0.9
        )
    elif args.optimization_method == "lora":
        if is_main_process:
            print(f"Setting up LoRA with rank={args.lora_rank}")
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            bias="none",
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model_for_optimizer = get_peft_model(model_for_optimizer, lora_config)
        # Re-wrap with DDP if needed
        if ddp_enabled:
            model = DDP(model_for_optimizer, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
            model_for_optimizer = model.module
        else:
            model = model_for_optimizer
        trainable_params = [p for p in model_for_optimizer.parameters() if p.requires_grad]
        optimizer = optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2)
        )
    else:
        raise ValueError(f"Unknown optimization_method: {args.optimization_method}")

    if is_main_process:
        print("Starting training...")
    best_val_loss = float('inf')
    global_step = 0
    if is_main_process:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    # Initialize training log (only on main process)
    log_file = os.path.join(output_dir, 'training_log.json') if is_main_process else None
    training_log = {
        'train_losses': [],
        'val_losses': [],
        'steps': [],
        'config': {
            'optimization_method': args.optimization_method,
            'learning_rate': args.learning_rate,
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'grad_accumulation_steps': args.grad_accumulation_steps,
            'lora_rank': args.lora_rank if args.optimization_method == 'lora' else None,
            'ddp_enabled': ddp_enabled,
            'world_size': world_size,
        }
    } if is_main_process else None

    for epoch in range(args.num_epochs):
        if is_main_process:
            print(f"\n--- Epoch {epoch+1}/{args.num_epochs} ---")
        if ddp_enabled:
            train_sampler.set_epoch(epoch)
        model.train()
        # Only show progress bar on main process
        train_iter = tqdm(train_loader, desc="Training") if is_main_process else train_loader
        for step, batch in enumerate(train_iter):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device_type_str, dtype=dtype):
                outputs = model(**batch)
                loss = outputs.loss
            loss = loss / args.grad_accumulation_steps
            loss.backward()
            if (step + 1) % args.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % args.log_interval == 0:
                    train_loss_value = loss.item() * args.grad_accumulation_steps
                    if is_main_process:
                        print(f"Step {global_step}: Train Loss = {train_loss_value:.4f}")
                        if training_log is not None:
                            training_log['train_losses'].append(train_loss_value)
                            training_log['steps'].append(global_step)
                if global_step % args.eval_interval == 0:
                    model.eval()
                    if is_main_process:
                        print("\nRunning validation...")
                    total_val_loss = 0
                    val_count = 0
                    with torch.no_grad():
                        val_iter = tqdm(val_loader, desc="Validating") if is_main_process else val_loader
                        for val_batch in val_iter:
                            val_batch = {k: v.to(device) for k, v in val_batch.items()}
                            with torch.autocast(device_type=device_type_str, dtype=dtype):
                                val_outputs = model(**val_batch)
                                val_loss = val_outputs.loss
                            total_val_loss += val_loss.item()
                            val_count += 1
                    # Aggregate validation loss across all processes
                    if ddp_enabled:
                        val_tensor = torch.tensor([total_val_loss, val_count], device=device)
                        dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
                        total_val_loss = val_tensor[0].item()
                        val_count = int(val_tensor[1].item())
                    avg_val_loss = total_val_loss / val_count
                    if is_main_process:
                        print(f"Step {global_step}: Validation Loss = {avg_val_loss:.4f}")
                        if training_log is not None:
                            training_log['val_losses'].append({
                                'step': global_step,
                                'loss': avg_val_loss
                            })
                            # Save log periodically (every eval_interval steps)
                            with open(log_file, 'w') as f:
                                json.dump(training_log, f, indent=2)
                            if is_main_process and global_step % (args.eval_interval * 10) == 0:
                                print(f"  -> Log saved at step {global_step}")
                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            if args.save_model:
                                print(f"  -> New best validation loss! Saving model to {output_dir}")
                                # Unwrap DDP for saving
                                model_to_save = model.module if ddp_enabled else model
                                model_to_save.save_pretrained(output_dir)
                                tokenizer.save_pretrained(output_dir)
                            else:
                                print(f"  -> New best validation loss: {avg_val_loss:.4f} (model not saved)")
                    model.train()
        
        # Save model at the end of each epoch if enabled
        if args.save_model and args.save_each_epoch and is_main_process:
            epoch_output_dir = os.path.join(output_dir, f'epoch_{epoch+1}')
            os.makedirs(epoch_output_dir, exist_ok=True)
            print(f"\nSaving model at end of epoch {epoch+1} to {epoch_output_dir}")
            model_to_save = model.module if ddp_enabled else model
            model_to_save.save_pretrained(epoch_output_dir)
            tokenizer.save_pretrained(epoch_output_dir)

    if is_main_process:
        print("\nTraining finished. Running one final evaluation...")
    model.eval()
    total_val_loss = 0
    val_count = 0
    with torch.no_grad():
        val_iter = tqdm(val_loader, desc="Final Validation") if is_main_process else val_loader
        for val_batch in val_iter:
            val_batch = {k: v.to(device) for k, v in val_batch.items()}
            with torch.autocast(device_type=device_type_str, dtype=dtype):
                val_outputs = model(**val_batch)
                val_loss = val_outputs.loss
            total_val_loss += val_loss.item()
            val_count += 1
    # Aggregate validation loss across all processes
    if ddp_enabled:
        val_tensor = torch.tensor([total_val_loss, val_count], device=device)
        dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
        total_val_loss = val_tensor[0].item()
        val_count = int(val_tensor[1].item())
    avg_val_loss = total_val_loss / val_count
    
    if is_main_process:
        print(f"Final Validation Loss = {avg_val_loss:.4f}")
        if training_log is not None:
            training_log['val_losses'].append({
                'step': global_step,
                'loss': avg_val_loss
            })
            training_log['final_val_loss'] = avg_val_loss
            training_log['best_val_loss'] = best_val_loss
            
            # Save final log
            with open(log_file, 'w') as f:
                json.dump(training_log, f, indent=2)
        
        if args.save_model:
            if avg_val_loss < best_val_loss:
                print(f"  -> Final model was the best! Saving model to {output_dir}")
                model_to_save = model.module if ddp_enabled else model
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
            else:
                print(f"  -> An earlier checkpoint was better (Val Loss: {best_val_loss:.4f}). Final model not saved.")
            print(f"\nProcess complete. Best model is saved in {output_dir}")
        else:
            print(f"\nProcess complete. Best validation loss: {best_val_loss:.4f} (model not saved)")
        print(f"Training log saved to {log_file}")
    
    # Clean up DDP
    if ddp_enabled:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()