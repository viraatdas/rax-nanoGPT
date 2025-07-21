"""
Training script for JAX/RAX nanoGPT
"""

import os
import time
import json
import math
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
import numpy as np

import jax
import jax.numpy as jnp
from jax import random, value_and_grad
import optax
from jaxtyping import Float, Int, PRNGKeyArray, Array

from model import GPTConfig, GPTParams, init_gpt_params, gpt_forward


@dataclass(frozen=True)
class TrainConfig:
    """Training configuration"""
    # Data
    dataset: str = "shakespeare"
    
    # Model
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 256
    bias: bool = False
    dropout: float = 0.2
    
    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    max_iters: int = 5000
    warmup_iters: int = 100
    lr_decay_iters: int = 5000
    min_lr: float = 1e-4
    weight_decay: float = 1e-1
    grad_clip: float = 1.0
    
    # Evaluation
    eval_interval: int = 500
    eval_iters: int = 200
    
    # Checkpointing
    out_dir: str = "out"
    checkpoint_interval: int = 1000
    
    # System
    seed: int = 1337
    
    # Logging
    log_interval: int = 10
    
    # Initialize from pretrained
    init_from: str = "scratch"  # 'scratch' or 'resume'


class DataLoader:
    """Simple data loader for training"""
    
    def __init__(self, data_path: str, block_size: int, batch_size: int):
        self.block_size = block_size
        self.batch_size = batch_size
        
        # Load data
        self.data = np.fromfile(data_path, dtype=np.uint16)
        print(f"Loaded {len(self.data):,} tokens from {data_path}")
        
    def get_batch(self, key: PRNGKeyArray) -> Tuple[Int[Array, "batch block_size"], Int[Array, "batch block_size"]]:
        """Get a random batch of data"""
        # Random starting positions
        key1, key2 = random.split(key)
        ix = random.randint(key1, (self.batch_size,), 0, len(self.data) - self.block_size - 1)
        
        # Gather sequences
        x = jnp.stack([self.data[i:i+self.block_size] for i in ix])
        y = jnp.stack([self.data[i+1:i+self.block_size+1] for i in ix])
        
        return x, y


def get_lr(step: int, config: TrainConfig) -> float:
    """Learning rate schedule with warmup and cosine decay"""
    # Warmup
    if step < config.warmup_iters:
        return config.learning_rate * step / config.warmup_iters
    
    # Cosine decay
    if step > config.lr_decay_iters:
        return config.min_lr
    
    decay_ratio = (step - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


def create_optimizer(config: TrainConfig) -> optax.GradientTransformation:
    """Create optimizer with AdamW"""
    # Create optimizer using config object
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adamw(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            b1=0.9,
            b2=0.95
        )
    )
    
    return optimizer


def train_step(
    params: GPTParams,
    opt_state: Any,
    x: Int[Array, "batch seq"],
    y: Int[Array, "batch seq"],
    key: PRNGKeyArray,
    model_config: GPTConfig,
    optimizer: optax.GradientTransformation
) -> Tuple[GPTParams, Any, Float[Array, ""], Dict[str, Float[Array, ""]]]:
    """Single training step"""
    
    def loss_fn(params):
        _, loss = gpt_forward(x, params, model_config, key, training=True, targets=y)
        return loss
    
    # Compute loss and gradients
    loss, grads = value_and_grad(loss_fn)(params)
    
    # Update parameters
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    # Compute gradient norm for logging
    grad_norm = optax.global_norm(grads)
    
    # Don't convert to float inside JIT - return JAX arrays
    metrics = {
        'loss': loss,
        'grad_norm': grad_norm
    }
    
    return params, opt_state, loss, metrics


def estimate_loss(
    params: GPTParams,
    model_config: GPTConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    eval_iters: int,
    key: PRNGKeyArray
) -> Dict[str, float]:
    """Estimate loss on train and validation sets"""
    out = {}
    
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = []
        keys = random.split(key, eval_iters + 1)
        key = keys[0]
        
        for i in range(eval_iters):
            x, y = loader.get_batch(keys[i+1])
            _, loss = gpt_forward(x, params, model_config, None, training=False, targets=y)
            losses.append(float(loss))
        
        out[split] = np.mean(losses)
    
    return out


def save_checkpoint(params: GPTParams, opt_state: Any, config: TrainConfig, 
                   model_config: GPTConfig, iter_num: int, best_val_loss: float) -> None:
    """Save model checkpoint"""
    os.makedirs(config.out_dir, exist_ok=True)
    checkpoint_path = os.path.join(config.out_dir, f'ckpt_{iter_num}.pkl')
    
    checkpoint = {
        'params': params,
        'opt_state': opt_state,
        'model_config': model_config,
        'train_config': config,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss
    }
    
    import pickle
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(path: str) -> Dict[str, Any]:
    """Load model checkpoint"""
    import pickle
    with open(path, 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint


def main():
    """Main training function"""
    import sys
    import importlib.util
    
    # Load configuration
    config_dict = {}
    
    # Get default values from TrainConfig
    default_config = TrainConfig()
    for field in default_config.__dataclass_fields__:
        config_dict[field] = getattr(default_config, field)
    
    # Override from config file if provided
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        print(f"Loading config from {config_file}")
        
        # Load the config module
        spec = importlib.util.spec_from_file_location("config", config_file)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # Update config dict with values from file
        for key in dir(config_module):
            if not key.startswith('_'):
                value = getattr(config_module, key)
                if key in config_dict:
                    config_dict[key] = value
                    print(f"  {key} = {value}")
    
    # Create frozen config instance
    config = TrainConfig(**config_dict)
    
    # Set random seed
    key = random.PRNGKey(config.seed)
    
    # Create output directory
    os.makedirs(config.out_dir, exist_ok=True)
    
    # Load dataset metadata
    data_dir = os.path.join('data', config.dataset)
    meta_path = os.path.join(data_dir, 'meta.json')
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    # Create model configuration
    model_config = GPTConfig(
        block_size=config.block_size,
        vocab_size=meta['vocab_size'],
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
        bias=config.bias
    )
    
    print(f"Model config: {model_config}")
    
    # Initialize model
    if config.init_from == 'scratch':
        print("Initializing model from scratch...")
        key, subkey = random.split(key)
        params = init_gpt_params(model_config, subkey)
        iter_num = 0
        best_val_loss = 1e9
    elif config.init_from == 'resume':
        print("Resuming from latest checkpoint...")
        # Find latest checkpoint
        ckpt_files = [f for f in os.listdir(config.out_dir) if f.startswith('ckpt_')]
        if not ckpt_files:
            raise ValueError("No checkpoints found to resume from")
        latest_ckpt = max(ckpt_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
        ckpt_path = os.path.join(config.out_dir, latest_ckpt)
        
        checkpoint = load_checkpoint(ckpt_path)
        params = checkpoint['params']
        model_config = checkpoint['model_config']
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed from iteration {iter_num}")
    
    # Create data loaders
    train_loader = DataLoader(
        os.path.join(data_dir, 'train.bin'),
        config.block_size,
        config.batch_size
    )
    val_loader = DataLoader(
        os.path.join(data_dir, 'val.bin'),
        config.block_size,
        config.batch_size
    )
    
    # Create initial optimizer
    optimizer = create_optimizer(config)
    opt_state = optimizer.init(params)
    
    # Training loop
    print(f"\nStarting training for {config.max_iters} iterations...")
    t0 = time.time()
    
    for iter_num in range(iter_num, config.max_iters):
        # Update learning rate
        current_lr = get_lr(iter_num, config)
        
        # Create optimizer with current learning rate
        current_optimizer = optax.chain(
            optax.clip_by_global_norm(config.grad_clip),
            optax.adamw(
                learning_rate=current_lr,
                weight_decay=config.weight_decay,
                b1=0.9,
                b2=0.95
            )
        )
        # Get batch
        key, subkey = random.split(key)
        x, y = train_loader.get_batch(subkey)
        
        # Training step
        key, subkey = random.split(key)
        params, opt_state, loss, metrics = train_step(
            params, opt_state, x, y, subkey, model_config, current_optimizer
        )
        
        # Logging
        if iter_num % config.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            # Convert JAX arrays to Python floats for logging
            loss_val = float(metrics['loss'])
            grad_norm_val = float(metrics['grad_norm'])
            print(f"iter {iter_num}: loss {loss_val:.4f}, "
                  f"grad_norm {grad_norm_val:.4f}, "
                  f"lr {current_lr:.2e}, time {dt*1000:.1f}ms")
        
        # Evaluation
        if iter_num % config.eval_interval == 0:
            key, subkey = random.split(key)
            losses = estimate_loss(params, model_config, train_loader, val_loader, 
                                 config.eval_iters, subkey)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, "
                  f"val loss {losses['val']:.4f}")
            
            # Save best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                if iter_num > 0:
                    save_checkpoint(params, opt_state, config, model_config, 
                                  iter_num, best_val_loss)
        
        # Regular checkpointing
        if iter_num % config.checkpoint_interval == 0 and iter_num > 0:
            save_checkpoint(params, opt_state, config, model_config, 
                          iter_num, best_val_loss)
    
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")
    
    # Save final checkpoint
    save_checkpoint(params, opt_state, config, model_config, 
                  config.max_iters, best_val_loss)


if __name__ == "__main__":
    main() 