"""
Simple test script to verify the JAX GPT model works
This tests the model without RAX tracing to verify basic functionality
"""

import jax
import jax.numpy as jnp
from jax import random

from model import GPTConfig, init_gpt_params, gpt_forward


def main():
    """Test the GPT model initialization and forward pass"""
    print("Testing JAX GPT model (without RAX tracing)...")
    
    # Initialize small config for testing
    config = GPTConfig(
        block_size=128,
        vocab_size=1024,  # Small vocab for testing
        n_layer=2,
        n_head=4,
        n_embd=128,
        dropout=0.1,
        bias=True
    )
    
    # Initialize random key
    key = random.PRNGKey(42)
    
    # Initialize parameters
    print(f"\nInitializing model with config: {config}")
    params = init_gpt_params(config, key)
    
    # Create test input
    batch_size = 2
    seq_len = 64
    idx = random.randint(key, (batch_size, seq_len), 0, config.vocab_size)
    
    # Test forward pass without targets
    print(f"\nTesting forward pass with input shape: {idx.shape}")
    logits, loss = gpt_forward(idx, params, config, key, training=False)
    print(f"Output logits shape: {logits.shape}")
    print(f"Loss (should be None): {loss}")
    
    # Test forward pass with targets (training mode)
    targets = random.randint(key, (batch_size, seq_len), 0, config.vocab_size)
    logits, loss = gpt_forward(idx, params, config, key, training=True, targets=targets)
    loss_val = float(loss)
    print(f"\nTraining mode - Loss: {loss_val:.4f}")
    
    # Test JIT compilation
    print("\nTesting JIT compilation...")
    # Create a wrapper that properly handles static arguments
    @jax.jit
    def jit_forward(idx, params, key, targets):
        return gpt_forward(idx, params, config, key, training=True, targets=targets)
    
    # Run JIT compiled version
    logits_jit, loss_jit = jit_forward(idx, params, key, targets)
    loss_jit_val = float(loss_jit)
    print(f"JIT compiled - Loss: {loss_jit_val:.4f}")
    
    # Verify outputs match
    print(f"Outputs match: {jnp.allclose(logits, logits_jit, rtol=1e-5)}")
    print(f"Losses match: {jnp.allclose(loss, loss_jit, rtol=1e-5)}")
    
    # Test generation (sampling)
    print("\nTesting generation...")
    gen_idx = jnp.array([[config.vocab_size // 2]])  # Start with a single token
    for _ in range(10):
        logits, _ = gpt_forward(gen_idx, params, config, key, training=False)
        # Get logits for last position
        last_logits = logits[0, -1, :]
        # Sample from logits (greedy for simplicity)
        next_token = jnp.argmax(last_logits)
        gen_idx = jnp.concatenate([gen_idx, next_token.reshape(1, 1)], axis=1)
    
    print(f"Generated sequence shape: {gen_idx.shape}")
    print(f"Generated tokens: {gen_idx[0].tolist()}")
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    main() 