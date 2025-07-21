# rax-nanoGPT

A minimal JAX/RAX implementation of GPT, inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).

[Rax](https://github.com/viraatdas/rax.git) provides a compile-time safe interface for JAX, enforcing memory safety and composable patterns in neural network model construction.

## Install

```bash
# Create virtual environment
uv venv --python 3.13
source .venv/bin/activate

# Install dependencies
uv add "rax @ git+https://github.com/viraatdas/rax.git" 
```

## Quick Start

```bash
# Prepare the Shakespeare dataset
cd data/shakespeare
python prepare.py
cd ../..

# Train a small GPT on Shakespeare
rax run train.py config/train_shakespeare.py

# Generate some text
rax run sample.py --out_dir=out-shakespeare --start="ROMEO:" --max_new_tokens=200
```

## What is RAX?

[RAX](https://github.com/viraatdas/rax) is a safe JAX frontend that adds compile-time shape validation and type safety. When you run with `rax run`, you get:

- **Shape validation**: Catch shape mismatches before training starts
- **Type safety**: All tensor dimensions are documented and checked
- **Automatic JIT**: RAX handles compilation automatically
- **Memory analysis**: Prevents OOM errors before they happen

Example of RAX catching errors:
```python
def bad_attention(q, k, v):  # No type hints
    scores = q @ k.T  # Shape mismatch discovered at runtime
    return softmax(scores) @ v

# With RAX type annotations:
def safe_attention(
    q: Float[Array, "batch heads seq_q dim"],
    k: Float[Array, "batch heads seq_k dim"],
    v: Float[Array, "batch heads seq_k dim"]
) -> Float[Array, "batch heads seq_q dim"]:
    # RAX validates shapes before execution
    scores = q @ k.swapaxes(-2, -1)
    return jax.nn.softmax(scores, axis=-1) @ v
```

## Performance
*Benchmarks coming soon*

## License

MIT
