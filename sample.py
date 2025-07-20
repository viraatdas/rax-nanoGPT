"""
Sample from a trained JAX/RAX GPT model
"""

import os
import pickle
from typing import Optional, List
import argparse

import jax
import jax.numpy as jnp
from jax import random
import tiktoken
from jaxtyping import Float, Int, PRNGKeyArray, Array

from model import GPTConfig, GPTParams, gpt_forward
# Import TrainConfig to handle pickle loading
from train import TrainConfig
import optax  # Needed for unpickling optimizer state


def sample_from_logits(
    logits: Float[Array, "vocab_size"],
    key: PRNGKeyArray,
    temperature: float = 1.0,
    top_k: Optional[int] = None
) -> Int[Array, ""]:
    """Sample from logits with temperature and optional top-k filtering"""
    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature
    
    # Apply top-k filtering
    if top_k is not None:
        # Get top k values and indices
        topk_logits, topk_indices = jax.lax.top_k(logits, top_k)
        # Create a new array with -inf for non-top-k values
        logits_filtered = jnp.full_like(logits, -jnp.inf)
        logits_filtered = logits_filtered.at[topk_indices].set(topk_logits)
        logits = logits_filtered
    
    # Sample from the distribution
    probs = jax.nn.softmax(logits)
    sample = random.categorical(key, logits=jnp.log(probs))
    
    return sample


def generate(
    params: GPTParams,
    config: GPTConfig,
    idx: Int[Array, "1 seq"],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    key: Optional[PRNGKeyArray] = None
) -> Int[Array, "1 seq+max_new_tokens"]:
    """Generate new tokens autoregressively"""
    # Initialize random key if not provided
    if key is None:
        key = random.PRNGKey(42)
    
    # Generate tokens one by one
    for _ in range(max_new_tokens):
        # Crop sequence if it's too long
        idx_cond = idx if idx.shape[1] <= config.block_size else idx[:, -config.block_size:]
        
        # Get predictions
        logits, _ = gpt_forward(idx_cond, params, config, None, training=False)
        
        # Focus on the last time step
        logits = logits[0, -1, :]  # (vocab_size,)
        
        # Sample from the distribution
        key, subkey = random.split(key)
        idx_next = sample_from_logits(logits, subkey, temperature, top_k)
        
        # Append sampled index to the running sequence
        idx = jnp.concatenate([idx, idx_next.reshape(1, 1)], axis=1)
    
    return idx


def main():
    parser = argparse.ArgumentParser(description='Sample from a trained GPT model')
    parser.add_argument('--out_dir', type=str, default='out', help='Checkpoint directory')
    parser.add_argument('--start', type=str, default='\n', help='Starting text')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to generate')
    parser.add_argument('--max_new_tokens', type=int, default=500, help='Number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=200, help='Top-k filtering')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Load the latest checkpoint
    ckpt_files = [f for f in os.listdir(args.out_dir) if f.startswith('ckpt_')]
    if not ckpt_files:
        raise ValueError(f"No checkpoints found in {args.out_dir}")
    
    latest_ckpt = max(ckpt_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
    ckpt_path = os.path.join(args.out_dir, latest_ckpt)
    
    print(f"Loading checkpoint from {ckpt_path}")
    with open(ckpt_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    params = checkpoint['params']
    model_config = checkpoint['model_config']
    print(f"Model config: {model_config}")
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # Encode the starting prompt
    if args.start.startswith('FILE:'):
        # Load prompt from file
        with open(args.start[5:], 'r') as f:
            start_text = f.read()
    else:
        start_text = args.start
    
    start_ids = enc.encode(start_text, allowed_special={'<|endoftext|>'})
    idx = jnp.array(start_ids).reshape(1, -1)
    
    print(f"Starting with: {repr(start_text)}")
    print(f"Encoded as {len(start_ids)} tokens")
    print("-" * 80)
    
    # Generate samples
    key = random.PRNGKey(args.seed)
    
    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"\n--- Sample {i+1} ---")
        
        # Generate
        key, subkey = random.split(key)
        generated = generate(
            params,
            model_config,
            idx,
            args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            key=subkey
        )
        
        # Decode and print
        generated_ids = generated[0].tolist()
        generated_text = enc.decode(generated_ids)
        print(generated_text)
        
        if args.num_samples > 1:
            print("-" * 80)


if __name__ == "__main__":
    main() 