"""Test script to verify RAX integration is working"""

import jax.numpy as jnp
from jaxtyping import Float, Array

def correct_add(x: Float[Array, "10"], y: Float[Array, "10"]) -> Float[Array, "10"]:
    """This should work correctly"""
    return x + y

def incorrect_dot(x: Float[Array, "32 100"], w: Float[Array, "128 64"]) -> Float[Array, "32 64"]:
    """This should fail with a shape mismatch error"""
    return jnp.dot(x, w)

# Test the correct function
print("Testing correct function...")
result = correct_add(jnp.ones(10), jnp.ones(10))
print(f"✓ correct_add result shape: {result.shape}")

# Test the incorrect function (should fail)
print("\nTesting incorrect function (should fail)...")
try:
    x = jnp.ones((32, 100))
    w = jnp.ones((128, 64))
    bad_result = incorrect_dot(x, w)
    print(f"✗ incorrect_dot unexpectedly succeeded with shape: {bad_result.shape}")
except Exception as e:
    print(f"✓ incorrect_dot correctly failed with: {type(e).__name__}: {e}") 