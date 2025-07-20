"""
Test script to verify the JAX/RAX GPT model
"""

from typing import Tuple, Optional
import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Float, Int, PRNGKeyArray, Array

from model import GPTConfig, GPTParams, init_gpt_params, gpt_forward


def run_forward_pass(
    idx: Int[Array, "batch seq"],
    params: GPTParams,
    config: GPTConfig,
    key: PRNGKeyArray,
    training: bool,
    targets: Optional[Int[Array, "batch seq"]] = None
) -> Tuple[Float[Array, "batch seq vocab_size"], Optional[Float[Array, ""]]]:
    """Run forward pass - this will be traced by RAX"""
    return gpt_forward(idx, params, config, key, training, targets)


def main() -> None:
    """Main test function - not traced by RAX"""
    print("Testing JAX/RAX GPT model...")
    
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
    logits, loss = run_forward_pass(idx, params, config, key, training=False)
    print(f"Output logits shape: {logits.shape}")
    print(f"Loss (should be None): {loss}")
    
    # Test forward pass with targets (training mode)
    targets = random.randint(key, (batch_size, seq_len), 0, config.vocab_size)
    logits, loss = run_forward_pass(idx, params, config, key, training=True, targets=targets)
    print(f"\nTraining mode - Loss: {loss}")
    
    # Test JIT compilation
    print("\nTesting JIT compilation...")
    jit_forward = jax.jit(run_forward_pass, static_argnames=['config', 'training'])
    
    # Run JIT compiled version
    logits_jit, loss_jit = jit_forward(idx, params, config, key, training=True, targets=targets)
    print(f"JIT compiled - Loss: {loss_jit}")
    
    # Verify outputs match
    print(f"Outputs match: {jnp.allclose(logits, logits_jit, rtol=1e-5)}")
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    main() 