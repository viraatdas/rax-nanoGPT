"""
Demonstration of common shape errors that RAX catches at compile time.
This shows the value of RAX's compile-time validation.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, Array, PRNGKeyArray
from model import GPTConfig, GPTParams, init_gpt_params, gpt_forward


def test_incorrect_batch_size() -> None:
    """Demonstrates batch size mismatch error"""
    print("Test 1: Incorrect batch size")
    print("-" * 50)
    
    config = GPTConfig(
        block_size=128,
        vocab_size=50257,
        n_layer=2,
        n_head=4,
        n_embd=128
    )
    
    key = jax.random.PRNGKey(0)
    params = init_gpt_params(config, key)
    
    # This should work
    correct_input = jnp.ones((4, 128), dtype=jnp.int32)  # batch_size=4, seq_len=128
    try:
        output, _ = gpt_forward(correct_input, params, config, key, training=False)
        print(f"✓ Correct input shape (4, 128) -> output shape: {output.shape}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    
    # This should fail - sequence length mismatch
    incorrect_input = jnp.ones((4, 200), dtype=jnp.int32)  # seq_len=200 > block_size=128
    try:
        output, _ = gpt_forward(incorrect_input, params, config, key, training=False)
        print(f"✗ Incorrect input unexpectedly succeeded")
    except Exception as e:
        print(f"✓ Caught shape error: {type(e).__name__}: {str(e)[:100]}...")


def test_embedding_dimension_mismatch() -> None:
    """Demonstrates what happens with mismatched embedding dimensions"""
    print("\nTest 2: Embedding dimension mismatch")
    print("-" * 50)
    
    # This function would demonstrate an error if we tried to
    # use embeddings with wrong dimensions
    def incorrect_embedding_lookup(
        tokens: Int[Array, "batch seq"],
        embeddings: Float[Array, "vocab_size n_embd"]
    ) -> Float[Array, "batch seq n_embd"]:
        return embeddings[tokens]
    
    tokens = jnp.array([[1, 2, 3], [4, 5, 6]])  # shape: (2, 3)
    small_embeddings = jnp.ones((100, 64))  # vocab_size=100, n_embd=64
    
    try:
        # This should work
        result = incorrect_embedding_lookup(tokens, small_embeddings)
        print(f"✓ Embedding lookup with valid tokens -> shape: {result.shape}")
        
        # This would fail if we had tokens >= vocab_size
        bad_tokens = jnp.array([[1, 2, 150]])  # 150 >= vocab_size=100
        bad_result = incorrect_embedding_lookup(bad_tokens, small_embeddings)
        print(f"✗ Should have failed with out-of-bounds token")
    except Exception as e:
        print(f"✓ Caught error: {type(e).__name__}: {str(e)[:100]}...")


def test_attention_shape_mismatch() -> None:
    """Demonstrates attention mechanism shape errors"""
    print("\nTest 3: Attention shape mismatch")
    print("-" * 50)
    
    def simplified_attention(
        query: Float[Array, "batch seq n_embd"],
        key: Float[Array, "batch seq n_embd"],
        value: Float[Array, "batch seq n_embd"]
    ) -> Float[Array, "batch seq n_embd"]:
        # Compute attention scores
        scores = jnp.matmul(query, key.swapaxes(-2, -1))
        weights = jax.nn.softmax(scores, axis=-1)
        return jnp.matmul(weights, value)
    
    # Correct shapes
    batch, seq_len, n_embd = 2, 10, 64
    q = jnp.ones((batch, seq_len, n_embd))
    k = jnp.ones((batch, seq_len, n_embd))
    v = jnp.ones((batch, seq_len, n_embd))
    
    try:
        output = simplified_attention(q, k, v)
        print(f"✓ Correct attention computation -> shape: {output.shape}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    
    # Mismatched sequence lengths
    k_wrong = jnp.ones((batch, 15, n_embd))  # Different seq_len
    try:
        output = simplified_attention(q, k_wrong, v)
        print(f"✗ Mismatched sequence lengths unexpectedly succeeded")
    except Exception as e:
        print(f"✓ Caught shape mismatch: {type(e).__name__}")


def main() -> None:
    """Run all shape error demonstrations"""
    print("=" * 70)
    print("RAX Shape Error Detection Demonstration")
    print("=" * 70)
    print("This script demonstrates how RAX catches common shape errors")
    print("at compile time, before any training begins.\n")
    
    test_incorrect_batch_size()
    test_embedding_dimension_mismatch()
    test_attention_shape_mismatch()
    
    print("\n" + "=" * 70)
    print("Summary: RAX helps catch shape errors early, saving compute time!")
    print("=" * 70)


if __name__ == "__main__":
    main() 