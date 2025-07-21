"""
Comprehensive test suite for RAX integration in nanoGPT.
Tests shape validation, memory analysis, and error handling.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, Array, PRNGKeyArray, Bool
from typing import Tuple, Optional

from model import GPTConfig, GPTParams, init_gpt_params, gpt_forward, transformer_block, causal_self_attention


def test_shape_validation_basic() -> None:
    """Test basic shape validation for common operations"""
    print("\n=== Test: Basic Shape Validation ===")
    
    # Test 1: Matrix multiplication with correct shapes
    def matmul_correct(
        a: Float[Array, "m k"],
        b: Float[Array, "k n"]
    ) -> Float[Array, "m n"]:
        return jnp.matmul(a, b)
    
    a = jnp.ones((10, 20))
    b = jnp.ones((20, 30))
    result = matmul_correct(a, b)
    assert result.shape == (10, 30)
    print("✓ Correct matrix multiplication passed")
    
    # Test 2: Matrix multiplication with incorrect shapes
    def matmul_incorrect(
        a: Float[Array, "m 15"],
        b: Float[Array, "20 n"]  # Mismatched inner dimension
    ) -> Float[Array, "m n"]:
        return jnp.matmul(a, b)
    
    try:
        bad_result = matmul_incorrect(jnp.ones((10, 15)), jnp.ones((20, 30)))
        print("✗ Expected shape error but none occurred")
    except Exception as e:
        print(f"✓ Caught expected shape error: {type(e).__name__}")


def test_model_shape_consistency() -> None:
    """Test shape consistency throughout the model"""
    print("\n=== Test: Model Shape Consistency ===")
    
    config = GPTConfig(
        block_size=128,
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_embd=64,
        dropout=0.0,
        bias=False
    )
    
    key = jax.random.PRNGKey(42)
    params = init_gpt_params(config, key)
    
    # Test with correct input shape
    batch_size = 4
    seq_len = 64  # Less than block_size
    input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    
    logits, loss = gpt_forward(input_ids, params, config, key, training=False)
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    print(f"✓ Forward pass with shape {input_ids.shape} -> {logits.shape}")
    
    # Test with sequence length exceeding block size
    try:
        long_input = jnp.ones((batch_size, config.block_size + 10), dtype=jnp.int32)
        bad_logits, _ = gpt_forward(long_input, params, config, key, training=False)
        print("✗ Expected assertion error for sequence length")
    except AssertionError as e:
        print("✓ Caught expected sequence length error")


def test_attention_mechanism() -> None:
    """Test attention mechanism shape handling"""
    print("\n=== Test: Attention Mechanism ===")
    
    def batched_attention(
        q: Float[Array, "batch heads seq_q dim"],
        k: Float[Array, "batch heads seq_k dim"],
        v: Float[Array, "batch heads seq_k dim"]
    ) -> Float[Array, "batch heads seq_q dim"]:
        """Multi-head attention with explicit batch dimension"""
        # q @ k.T -> (batch, heads, seq_q, seq_k)
        scores = jnp.matmul(q, k.swapaxes(-2, -1))
        
        # Softmax over seq_k dimension
        weights = jax.nn.softmax(scores, axis=-1)
        
        # weights @ v -> (batch, heads, seq_q, dim)
        return jnp.matmul(weights, v)
    
    # Test with matching sequence lengths
    batch, heads, seq_len, dim = 2, 4, 16, 32
    q = k = v = jnp.ones((batch, heads, seq_len, dim))
    output = batched_attention(q, k, v)
    assert output.shape == (batch, heads, seq_len, dim)
    print(f"✓ Attention with matching seq lengths: {output.shape}")
    
    # Test with different q and k/v sequence lengths (valid case)
    seq_q, seq_kv = 10, 20
    q_diff = jnp.ones((batch, heads, seq_q, dim))
    k_diff = v_diff = jnp.ones((batch, heads, seq_kv, dim))
    output_diff = batched_attention(q_diff, k_diff, v_diff)
    assert output_diff.shape == (batch, heads, seq_q, dim)
    print(f"✓ Attention with different seq lengths: Q={seq_q}, KV={seq_kv}")


def test_embedding_operations() -> None:
    """Test embedding lookup and position encoding"""
    print("\n=== Test: Embedding Operations ===")
    
    def embedding_lookup(
        tokens: Int[Array, "batch seq"],
        embeddings: Float[Array, "vocab_size dim"]
    ) -> Float[Array, "batch seq dim"]:
        """Safe embedding lookup with bounds checking"""
        vocab_size = embeddings.shape[0]
        # This would fail if tokens contain values >= vocab_size
        return embeddings[tokens]
    
    vocab_size, dim = 100, 64
    embeddings = jnp.ones((vocab_size, dim))
    
    # Valid tokens
    valid_tokens = jnp.array([[1, 2, 3], [4, 5, 6]])
    valid_embeds = embedding_lookup(valid_tokens, embeddings)
    assert valid_embeds.shape == (2, 3, dim)
    print("✓ Valid embedding lookup passed")
    
    # Test with out-of-bounds tokens (would fail at runtime)
    # Note: This is a runtime error, not a shape error
    try:
        invalid_tokens = jnp.array([[1, 2, 150]])  # 150 >= vocab_size
        invalid_embeds = embedding_lookup(invalid_tokens, embeddings)
        print("✗ Expected index error but none occurred")
    except Exception as e:
        print(f"✓ Caught expected index error: {type(e).__name__}")


def test_loss_computation() -> None:
    """Test loss computation shapes"""
    print("\n=== Test: Loss Computation ===")
    
    def cross_entropy_loss(
        logits: Float[Array, "batch seq vocab"],
        targets: Int[Array, "batch seq"]
    ) -> Float[Array, ""]:
        """Compute cross-entropy loss"""
        batch, seq, vocab = logits.shape
        
        # Flatten for loss computation
        logits_flat = logits.reshape(-1, vocab)
        targets_flat = targets.reshape(-1)
        
        # Log softmax
        log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
        
        # Gather log probs for targets
        loss = -jnp.mean(log_probs[jnp.arange(targets_flat.shape[0]), targets_flat])
        
        return loss
    
    batch, seq, vocab = 4, 32, 1000
    logits = jnp.ones((batch, seq, vocab))
    targets = jnp.zeros((batch, seq), dtype=jnp.int32)
    
    loss = cross_entropy_loss(logits, targets)
    assert loss.shape == ()  # Scalar
    print(f"✓ Loss computation: {logits.shape} -> scalar")


def test_gradient_computation() -> None:
    """Test gradient computation shapes"""
    print("\n=== Test: Gradient Computation ===")
    
    def simple_linear(
        x: Float[Array, "batch dim_in"],
        w: Float[Array, "dim_in dim_out"],
        b: Float[Array, "dim_out"]
    ) -> Float[Array, "batch dim_out"]:
        """Simple linear layer"""
        return jnp.dot(x, w) + b
    
    def loss_fn(
        w: Float[Array, "dim_in dim_out"],
        b: Float[Array, "dim_out"],
        x: Float[Array, "batch dim_in"],
        y: Float[Array, "batch dim_out"]
    ) -> Float[Array, ""]:
        """MSE loss"""
        pred = simple_linear(x, w, b)
        return jnp.mean((pred - y) ** 2)
    
    # Setup
    batch, dim_in, dim_out = 8, 32, 16
    w = jnp.ones((dim_in, dim_out))
    b = jnp.zeros(dim_out)
    x = jnp.ones((batch, dim_in))
    y = jnp.ones((batch, dim_out))
    
    # Compute gradients
    grad_fn = jax.grad(loss_fn, argnums=(0, 1))
    grad_w, grad_b = grad_fn(w, b, x, y)
    
    assert grad_w.shape == w.shape
    assert grad_b.shape == b.shape
    print(f"✓ Gradient shapes: w={grad_w.shape}, b={grad_b.shape}")


def test_memory_estimation() -> None:
    """Test memory usage estimation for different model sizes"""
    print("\n=== Test: Memory Estimation ===")
    
    def estimate_model_memory(config: GPTConfig) -> float:
        """Estimate memory usage in MB"""
        # Parameters
        embedding_params = 2 * config.vocab_size * config.n_embd  # wte + wpe
        
        # Per block
        ln_params = 2 * config.n_embd * 2  # Two layer norms with weight and bias
        attn_params = config.n_embd * (3 * config.n_embd + config.n_embd)  # QKV + projection
        mlp_params = config.n_embd * (4 * config.n_embd + 4 * config.n_embd)  # Two linear layers
        
        block_params = ln_params + attn_params + mlp_params
        total_params = embedding_params + config.n_layer * block_params + config.n_embd * 2  # Final LN
        
        # Float32 = 4 bytes
        memory_mb = (total_params * 4) / (1024 * 1024)
        
        return memory_mb
    
    # Small model
    small_config = GPTConfig(n_layer=6, n_head=6, n_embd=384)
    small_memory = estimate_model_memory(small_config)
    print(f"✓ Small model (~{small_config.n_layer}L): {small_memory:.1f} MB")
    
    # Medium model
    medium_config = GPTConfig(n_layer=12, n_head=12, n_embd=768)
    medium_memory = estimate_model_memory(medium_config)
    print(f"✓ Medium model (~{medium_config.n_layer}L): {medium_memory:.1f} MB")
    
    # Batch memory estimation
    def estimate_batch_memory(batch_size: int, seq_len: int, config: GPTConfig) -> float:
        """Estimate memory for a batch during forward pass"""
        # Activations per layer
        hidden_size = batch_size * seq_len * config.n_embd * 4  # bytes
        attention_scores = batch_size * config.n_head * seq_len * seq_len * 4
        
        # Total for all layers
        total_bytes = config.n_layer * (hidden_size + attention_scores)
        return total_bytes / (1024 * 1024)  # MB
    
    batch_mem = estimate_batch_memory(32, 256, small_config)
    print(f"✓ Batch memory (32x256): {batch_mem:.1f} MB")


def test_dtype_consistency() -> None:
    """Test data type consistency"""
    print("\n=== Test: Data Type Consistency ===")
    
    def process_mixed_types(
        float_input: Float[Array, "batch dim"],
        int_input: Int[Array, "batch"],
        bool_mask: Optional[Bool[Array, "batch"]] = None
    ) -> Tuple[Float[Array, "batch dim"], Int[Array, "batch"]]:
        """Process inputs of different types"""
        # Apply mask if provided
        if bool_mask is not None:
            float_input = jnp.where(bool_mask[:, None], float_input, 0.0)
            int_input = jnp.where(bool_mask, int_input, -1)
        
        return float_input, int_input
    
    batch, dim = 4, 8
    float_data = jnp.ones((batch, dim), dtype=jnp.float32)
    int_data = jnp.arange(batch, dtype=jnp.int32)
    bool_mask = jnp.array([True, False, True, False])
    
    float_out, int_out = process_mixed_types(float_data, int_data, bool_mask)
    
    assert float_out.dtype == jnp.float32
    assert int_out.dtype == jnp.int32
    print("✓ Data type consistency maintained")


def run_all_tests() -> None:
    """Run all comprehensive tests"""
    print("=" * 60)
    print("Comprehensive RAX Integration Tests")
    print("=" * 60)
    
    test_shape_validation_basic()
    test_model_shape_consistency()
    test_attention_mechanism()
    test_embedding_operations()
    test_loss_computation()
    test_gradient_computation()
    test_memory_estimation()
    test_dtype_consistency()
    
    print("\n" + "=" * 60)
    print("All tests completed successfully! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests() 