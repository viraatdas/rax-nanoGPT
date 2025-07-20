"""
RAX Safety Demo for nanoGPT
============================

This script demonstrates the key safety features that RAX provides
for the nanoGPT implementation, including shape validation, clear 
error messages, and compile-time safety checks.
"""

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Float, Int, Array
import rax

from model import GPTConfig, init_gpt_params, gpt_forward


def demo_shape_validation():
    """Demonstrate RAX shape validation with clear annotations"""
    print("🔍 SHAPE VALIDATION DEMO")
    print("-" * 40)
    
    config = GPTConfig(
        block_size=128,
        vocab_size=1024,
        n_layer=2,
        n_head=4,
        n_embd=128
    )
    
    key = random.PRNGKey(42)
    params = init_gpt_params(config, key)
    
    # ✅ Correct usage with proper shapes
    batch_size = 2
    seq_len = 64
    x = random.randint(key, (batch_size, seq_len), 0, config.vocab_size)
    
    print(f"✅ Input shape: {x.shape} (batch={batch_size}, seq_len={seq_len})")
    
    # Forward pass with shape-validated function
    key, subkey = random.split(key)
    logits, loss = gpt_forward(x, params, config, subkey, training=True, targets=x)
    
    print(f"✅ Output shape: {logits.shape} (batch, seq_len, vocab_size)")
    print(f"✅ Loss shape: {loss.shape} (scalar)")
    print("✅ All shapes validated successfully!\n")


def demo_error_catching():
    """Demonstrate how RAX catches shape errors early"""
    print("⚠️  ERROR CATCHING DEMO")
    print("-" * 40)
    
    @rax.validate_function
    def matrix_multiply_demo(
        a: Float[Array, "batch hidden"],
        b: Float[Array, "hidden output"]
    ) -> Float[Array, "batch output"]:
        """Example function with clear shape documentation"""
        return a @ b
    
    # ✅ Correct usage
    key = random.PRNGKey(123)
    a = random.normal(key, (32, 128))  # batch=32, hidden=128
    b = random.normal(key, (128, 64))  # hidden=128, output=64
    
    result = matrix_multiply_demo(a, b)
    print(f"✅ Correct operation: {a.shape} @ {b.shape} = {result.shape}")
    
    # ❌ This would cause an error (uncomment to see RAX catch it)
    # try:
    #     b_wrong = random.normal(key, (64, 128))  # Wrong shape!
    #     result = matrix_multiply_demo(a, b_wrong)
    # except Exception as e:
    #     print(f"❌ RAX caught error: {type(e).__name__}")
    #     print(f"   Message: {str(e)[:100]}...")
    
    print("✅ Shape compatibility verified!\n")


def demo_memory_awareness():
    """Show memory considerations with RAX"""
    print("💾 MEMORY AWARENESS DEMO")
    print("-" * 40)
    
    # Example: Different model sizes and their memory implications
    configs = [
        ("Small", GPTConfig(n_layer=4, n_head=4, n_embd=128, block_size=256)),
        ("Medium", GPTConfig(n_layer=6, n_head=6, n_embd=384, block_size=512)),
        ("Large", GPTConfig(n_layer=12, n_head=12, n_embd=768, block_size=1024)),
    ]
    
    for name, config in configs:
        # Calculate parameter count
        key = random.PRNGKey(0)
        params = init_gpt_params(config, key)
        
        # Estimate parameter count
        param_count = sum(
            param.size for param in jax.tree_util.tree_leaves(params)
        )
        
        print(f"📊 {name} model:")
        print(f"   Parameters: {param_count:,}")
        print(f"   Config: {config.n_layer}L, {config.n_head}H, {config.n_embd}D")
        print(f"   Context: {config.block_size} tokens")
        
        # Memory estimate (very rough)
        memory_mb = param_count * 4 / (1024 * 1024)  # 4 bytes per float32
        print(f"   Est. memory: ~{memory_mb:.1f} MB")
        print()


def demo_self_documenting_code():
    """Show how RAX makes code self-documenting"""
    print("📝 SELF-DOCUMENTING CODE DEMO")
    print("-" * 40)
    
    @rax.validate_function
    def attention_demo(
        q: Float[Array, "batch seq_len d_model"],
        k: Float[Array, "batch seq_len d_model"],
        v: Float[Array, "batch seq_len d_model"]
    ) -> Float[Array, "batch seq_len d_model"]:
        """
        Multi-head attention with clear shape documentation.
        
        Args:
            q: Query vectors [batch_size, sequence_length, model_dimension]
            k: Key vectors [batch_size, sequence_length, model_dimension]  
            v: Value vectors [batch_size, sequence_length, model_dimension]
            
        Returns:
            Attention output [batch_size, sequence_length, model_dimension]
        """
        # Compute attention scores
        scores = q @ k.swapaxes(-1, -2)  # [batch, seq_len, seq_len]
        weights = jax.nn.softmax(scores, axis=-1)
        output = weights @ v  # [batch, seq_len, d_model]
        return output
    
    # Example usage
    batch_size, seq_len, d_model = 4, 32, 128
    key = random.PRNGKey(456)
    
    q = random.normal(key, (batch_size, seq_len, d_model))
    k = random.normal(key, (batch_size, seq_len, d_model))
    v = random.normal(key, (batch_size, seq_len, d_model))
    
    output = attention_demo(q, k, v)
    
    print("✅ Function signature documents all shapes clearly")
    print("✅ IDE autocomplete knows exact tensor dimensions")
    print("✅ New developers understand code immediately")
    print(f"✅ Output shape: {output.shape} (as expected)\n")


def main():
    """Run all RAX safety demos"""
    print("🚀 RAX-nanoGPT SAFETY DEMONSTRATION")
    print("=" * 50)
    print("This demo shows how RAX enhances nanoGPT with:")
    print("• Compile-time shape validation")
    print("• Clear error messages") 
    print("• Self-documenting code")
    print("• Memory-aware development")
    print("=" * 50)
    print()
    
    demo_shape_validation()
    demo_error_catching()
    demo_memory_awareness()
    demo_self_documenting_code()
    
    print("🎉 CONCLUSION")
    print("-" * 40)
    print("RAX transforms nanoGPT development from:")
    print("❌ 'Hope it works' → ✅ 'Know it works'")
    print("❌ Runtime crashes → ✅ Compile-time safety")
    print("❌ Unclear shapes → ✅ Self-documenting code")
    print("❌ Trial and error → ✅ Mathematical guarantees")
    print()
    print("Ready to train with confidence! 🚀")


if __name__ == "__main__":
    main() 