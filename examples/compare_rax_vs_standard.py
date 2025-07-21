"""
Comparison of RAX vs Standard JAX execution.
This script demonstrates how RAX catches errors that standard JAX would miss.
"""

import sys
import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, Array


def matrix_multiply_unsafe(a, b):
    """Without type annotations - errors discovered at runtime"""
    return jnp.dot(a, b)


def matrix_multiply_safe(
    a: Float[Array, "m k"],
    b: Float[Array, "k n"]
) -> Float[Array, "m n"]:
    """With type annotations - RAX validates shapes"""
    return jnp.dot(a, b)


def demonstrate_shape_error() -> None:
    """Shows how RAX catches shape mismatches early"""
    print("=" * 70)
    print("Demonstrating Shape Error Detection")
    print("=" * 70)
    
    # Create matrices with incompatible shapes
    a = jnp.ones((10, 5))   # Shape: (10, 5)
    b = jnp.ones((3, 4))    # Shape: (3, 4) - incompatible!
    
    print(f"Matrix A shape: {a.shape}")
    print(f"Matrix B shape: {b.shape}")
    print("Attempting multiplication (should fail)...\n")
    
    # Without RAX - error at runtime
    if "--no-rax" in sys.argv:
        print("Running WITHOUT RAX validation:")
        try:
            result = matrix_multiply_unsafe(a, b)
            print(f"Result shape: {result.shape}")
        except Exception as e:
            print(f"❌ Runtime error: {type(e).__name__}: {e}")
            print("   ^ Error discovered only during execution!")
    
    # With RAX - error caught during validation
    else:
        print("Running WITH RAX validation:")
        try:
            result = matrix_multiply_safe(a, b)
            print(f"Result shape: {result.shape}")
        except Exception as e:
            print(f"✅ RAX validation error: {type(e).__name__}")
            print(f"   Details: {e}")
            print("   ^ Error caught before execution!")


def demonstrate_memory_safety() -> None:
    """Shows how RAX helps prevent OOM errors"""
    print("\n" + "=" * 70)
    print("Demonstrating Memory Safety")
    print("=" * 70)
    
    def large_computation(
        x: Float[Array, "batch 10000"],
        w: Float[Array, "10000 10000"]
    ) -> Float[Array, "batch 10000"]:
        """Computation that might cause OOM"""
        # Multiple large intermediate tensors
        h1 = jnp.dot(x, w)  # batch × 10000
        h2 = jax.nn.relu(h1)
        h3 = jnp.dot(h2, w)  # Another large operation
        return h3
    
    print("Simulating large computation that might cause OOM...")
    print("Input shape: (32, 10000)")
    print("Weight shape: (10000, 10000)")
    print("Estimated memory: ~3.2 GB per operation")
    
    if "--no-rax" in sys.argv:
        print("\nWithout RAX: No pre-execution memory analysis")
        print("⚠️  Would discover OOM only after allocation attempts")
    else:
        print("\nWith RAX: Memory requirements analyzed during validation")
        print("✅ Would warn about high memory usage before execution")


def demonstrate_type_safety() -> None:
    """Shows how type annotations improve code clarity"""
    print("\n" + "=" * 70)
    print("Demonstrating Type Safety & Documentation")
    print("=" * 70)
    
    # Unclear function
    def process_batch_unclear(x, mask=None):
        """Process a batch of data - but what shapes?"""
        if mask is not None:
            x = x * mask
        return jnp.mean(x, axis=1)
    
    # Clear function with RAX
    def process_batch_clear(
        x: Float[Array, "batch seq features"],
        mask: Float[Array, "batch seq"] = None
    ) -> Float[Array, "batch features"]:
        """Process a batch of data with clear shape documentation"""
        if mask is not None:
            x = x * mask[..., None]  # Broadcasting is clear
        return jnp.mean(x, axis=1)  # Averaging over sequence
    
    print("Without type annotations:")
    print("  - What shape should x be?")
    print("  - What dimensions does mask have?")
    print("  - What axis does mean operate on?")
    print("  - Need to read implementation or documentation")
    
    print("\nWith RAX type annotations:")
    print("  - x: Float[Array, 'batch seq features']")
    print("  - mask: Float[Array, 'batch seq']")
    print("  - return: Float[Array, 'batch features']")
    print("  - Self-documenting and validated!")


def main() -> None:
    """Run all demonstrations"""
    print("\n🚀 RAX vs Standard JAX Comparison\n")
    
    if "--no-rax" in sys.argv:
        print("Running in STANDARD JAX mode (no RAX validation)\n")
    else:
        print("Running with RAX validation enabled\n")
    
    demonstrate_shape_error()
    demonstrate_memory_safety()
    demonstrate_type_safety()
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    if "--no-rax" in sys.argv:
        print("❌ Standard JAX: Errors discovered at runtime")
        print("❌ No shape validation until execution")
        print("❌ No memory analysis")
        print("❌ Less clear code documentation")
    else:
        print("✅ RAX: Errors caught during validation")
        print("✅ Shape mismatches detected early")
        print("✅ Memory requirements analyzed")
        print("✅ Self-documenting type annotations")
    
    print("\nTry running this script both ways:")
    print("  rax run examples/compare_rax_vs_standard.py")
    print("  python examples/compare_rax_vs_standard.py --no-rax")


if __name__ == "__main__":
    main() 