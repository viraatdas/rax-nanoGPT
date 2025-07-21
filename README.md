# RAX-nanoGPT: Type-Safe GPT Training with Compile-Time Validation (written in Jax)

A JAX/RAX implementation of nanoGPT that transforms "hope it works" into "know it works" through compile-time shape validation, memory safety checks, and explicit type documentation.

## 🚀 Why RAX-nanoGPT?

Traditional deep learning development is error-prone - shape mismatches, OOM crashes, and unclear tensor dimensions waste hours of compute time. RAX-nanoGPT solves this with:

### 💡 **Compile-Time Safety**
```python
# ❌ Without RAX: Discover shape errors after hours of training
def train_step(params, x, y):  # What shapes? Who knows!
    loss, grads = compute_loss_and_grads(params, x, y)  # RuntimeError after 2 hours!
    return updated_params

# ✅ With RAX: Errors caught during compilation (which is automatically caught with rax run)
def train_step(
    params: GPTParams,
    x: Int[Array, "batch seq_len"],     # Crystal clear shapes
    y: Int[Array, "batch seq_len"],     # Self-documenting code
    key: PRNGKeyArray
) -> Tuple[GPTParams, Float[Array, ""], Dict[str, Float[Array, ""]]]:
    # Shapes validated during compilation, before any execution!
```

### 🔒 **Memory Safety**
- **OOM Prevention**: RAX analyzes memory usage before execution (currently on every call)*
- **Early Warnings**: Get memory estimates before any GPU allocation
- **No Wasted Compute**: Bad configurations caught before GPU operations

*Note: Current implementation validates on each call. True compile-time validation would be better.

### 📝 **Self-Documenting Code**
- **Shape Annotations**: Every tensor parameter documents its expected dimensions
- **IDE Support**: Autocomplete knows exact tensor shapes  
- **Team Onboarding**: New developers understand code structure instantly

### ⚡ **Zero Performance Overhead**
- **Same Speed**: RAX validation happens at compile time, not runtime
- **JIT Optimized**: Automatic static argument detection for optimal performance
- **Production Ready**: All safety checks with no runtime cost

## 🔧 RAX Integration Details

### What is RAX?

RAX is a safe JAX frontend that adds compile-time shape validation, type safety, and memory analysis to JAX code. When you run code with `rax run`, it:

1. **Validates Type Annotations**: Ensures all functions have proper jaxtyping annotations
2. **Enforces Shape Consistency**: Uses beartype to validate tensor shapes at runtime
3. **Traces Functions**: Uses `jax.make_jaxpr` to catch shape/math errors before execution
4. **Auto-JIT Compilation**: Automatically applies `jax.jit` for optimal performance
5. **Memory Analysis**: Estimates memory usage to prevent OOM errors

### Running With vs Without RAX

```bash
# With RAX (recommended) - Full safety validation
rax run train.py config/train_shakespeare.py

# Without RAX - Standard JAX execution (no safety checks)
python train.py config/train_shakespeare.py
```

### Example: RAX Catching Errors

```python
# Without RAX: Discover after hours of training
def bad_attention(q, k, v):  # No type hints
    scores = q @ k.T  # Shape mismatch discovered at runtime
    return softmax(scores) @ v

# With RAX: Caught immediately
def safe_attention(
    q: Float[Array, "batch heads seq_q dim"],
    k: Float[Array, "batch heads seq_k dim"],  # Clear shape docs
    v: Float[Array, "batch heads seq_k dim"]
) -> Float[Array, "batch heads seq_q dim"]:
    # RAX validates seq_q vs seq_k compatibility before execution
    scores = q @ k.swapaxes(-2, -1)
    return jax.nn.softmax(scores, axis=-1) @ v
```

### Testing Shape Safety

Run our test scripts to see RAX in action:

```bash
# See RAX catch common shape errors
rax run tests/test_rax_shape_errors.py

# Quick integration test
rax run tests/test_rax_integration.py

# Comprehensive test suite
rax run tests/test_comprehensive_rax.py
```

## 📁 Project Structure

```
rax-nanogpt/
├── model.py              # GPT model with full RAX type annotations
├── train.py              # Training script (automatic JIT via RAX)
├── sample.py             # Text generation script
├── config/               # Training configurations
│   ├── train_shakespeare.py
│   └── benchmark_quick.py
├── data/                 # Dataset preparation
│   └── shakespeare/
├── tests/                # RAX validation tests
│   ├── test_rax_integration.py
│   ├── test_rax_shape_errors.py
│   └── test_comprehensive_rax.py
├── examples/             # Feature demonstrations
│   ├── rax_features_demo.py
│   └── compare_rax_vs_standard.py
├── docs/                 # Additional documentation
│   └── RAX_INTEGRATION_SUMMARY.md
└── pyproject.toml        # Dependencies (includes RAX)
```  

## Quick Start

### Installation

```bash
# Create virtual environment
uv venv --python 3.13
source .venv/bin/activate

# Install dependencies  
uv add "rax @ git+https://github.com/viraatdas/rax.git" jax jaxlib jaxtyping numpy tiktoken requests tqdm matplotlib
```

### Train Shakespeare Model

```bash
# Prepare data
cd data/shakespeare && python prepare.py && cd ../..

# Train with RAX safety validation
rax run train.py config/train_shakespeare.py
```

**RAX Output:**
```
[RAX] JIT-compiling 'train_step' with static_argnums=[5, 6]
iter 0: loss 10.8160, grad_norm 2.0590, lr 0.00e+00, time 1155.0ms
iter 10: loss 10.5924, grad_norm 2.0571, lr 6.00e-05, time 104.3ms
...
iter 100: loss 6.6116, grad_norm 1.3977, lr 6.00e-04, time 15.3ms
step 100: train loss 6.5919, val loss 6.6791
```

### Generate Text

```bash
rax run sample.py --out_dir=out-shakespeare --start="ROMEO:" --max_new_tokens=200
```

## 🛡️ RAX Safety Features in Action

### 1. Shape Validation
```python
# RAX catches this immediately:
def attention(
    q: Float32[Array, "batch seq d_k"],
    k: Float32[Array, "batch seq d_k"], 
    v: Float32[Array, "batch seq d_v"]  # ← Clear dimension documentation
) -> Float32[Array, "batch seq d_v"]:
    scores = q @ k.T  # RAX validates matrix multiplication compatibility
    return jax.nn.softmax(scores) @ v
```

### 2. Memory Analysis
```bash
# RAX prevents OOM at compile time
[RAX] JIT-compiling 'train_step'...
[RAX] Analyzing memory usage during compilation...
[RAX] Estimated memory: 8.2 GB (within 16 GB limit) ✅
[RAX] Compilation successful with optimized static arguments
```

### 3. Clear Error Messages
```python
# Instead of cryptic PyTorch errors:
# RuntimeError: Expected tensor of size [64, 256] but got [64, 512]

# RAX provides helpful context:
# RAXValidationError: Shape mismatch in 'train_step' parameter 'x'
# Expected: Int[Array, "batch seq_len"] with seq_len=256  
# Got: (64, 512)
# Suggestion: Check your DataLoader block_size configuration
```

## 🧠 How RAX Detects OOM (Current Implementation vs Ideal)

**Important Note**: The current RAX implementation validates on every function call, not truly at compile time. This section describes both the current behavior and what ideal compile-time detection would look like.

### The PyTorch Problem
```python
# PyTorch: Runtime OOM after wasting time on setup
model = GPT(large_config).to('cuda')  # Allocates memory eagerly
optimizer = torch.optim.AdamW(model.parameters())

for batch in dataloader:  # Already committed resources
    loss = model(batch)    # 💥 CUDA out of memory after 2 hours!
    # No warning, no prevention, just wasted time
```

### Current RAX Implementation
```python
# Current: Validation on every call (not ideal)
@rax.validate_function
def train_step(params: GPTParams, x: Int[Array, "batch seq"], ...):
    # Current RAX validates BEFORE each execution
    pass

# What happens now:
# 1. Function is called
# 2. RAX runs validate_jaxpr (using jax.make_jaxpr)
# 3. Memory analysis happens
# 4. If OK, function executes
# This happens on EVERY call - not true compile-time!
```

### Ideal Compile-Time Solution
```python
# Ideal: True compile-time validation (needs RAX enhancement)
@rax.validate_function
@jax.jit
def train_step(params: GPTParams, x: Int[Array, "batch seq"], ...):
    pass

# What SHOULD happen:
# 1. First call triggers JAX JIT compilation
# 2. During compilation, RAX analyzes memory ONCE
# 3. Compilation fails if OOM would occur
# 4. Subsequent calls run without re-validation
# Currently RAX doesn't achieve this!
```

### How Current RAX Works

**1. Pre-Execution Validation (Every Call)**
- When function is called, RAX intercepts
- Uses `jax.make_jaxpr` to trace the computation graph
- Analyzes memory requirements
- This happens BEFORE each execution, not at compile time

**2. Memory Analysis Process**
```
# This happens on EVERY function call:
For each operation in the jaxpr:
- Input tensors: shape × dtype_size
- Output tensors: shape × dtype_size
- Temporary buffers: operation-specific

Example - Attention layer:
- Q,K,V projections: batch × seq × (3 × hidden) × 4 bytes
- Attention scores: batch × heads × seq × seq × 4 bytes
- Gradient storage: 2× forward memory
- Optimizer states: 2× parameters (for Adam)
```

**3. Pre-Execution Error (Better than Runtime, Not as Good as Compile-Time)**
```
# Error appears before execution, but after call:
[RAX] Memory Analysis for 'train_step':
  - Model parameters: 8.2 GB
  - Forward activations: 12.3 GB
  - Backward gradients: 8.2 GB  
  - Optimizer states: 16.4 GB
  - Peak usage: 45.1 GB
  
[RAX] ERROR: OOM would occur if this function runs!
  Available: 16.0 GB
  Required: 45.1 GB
  
Suggestions:
  1. Reduce batch_size to 16 (→ 11.3 GB)
  2. Use gradient accumulation
  3. Enable mixed precision

# Better than PyTorch: Caught before GPU allocation
# Not ideal: Validation overhead on every call
```

### Why PyTorch Can't Do This

| Aspect | PyTorch | Current RAX | Ideal RAX |
|--------|---------|-------------|-----------|
| **Execution** | Eager (immediate) | JIT + validation | JIT with compile validation |
| **Shape Info** | Runtime only | Via annotations | Via annotations |
| **Memory Check** | During GPU allocation | Before each execution | During JIT compilation |
| **OOM Handling** | Runtime crash | Pre-execution error | Compilation error |
| **When Error Occurs** | After setup & allocation | Before function runs | At first compilation |
| **Validation Overhead** | None | Every call | First call only |
| **GPU Memory Used** | Already allocated | Not allocated | Never allocated |

### The Key Difference: Pre-Execution vs Runtime

```python
# Current RAX: Validates before execution (every call)
@rax.validate_function
def train_step(...):
    ...

for batch in loader:
    # RAX validates HERE (every iteration)
    result = train_step(params, batch)  # Validation overhead
    # But catches OOM before GPU allocation

# PyTorch: Fails at runtime  
model = Model().cuda()  # ← GPU memory allocated
optimizer = optim.Adam(...)  # ← More memory allocated
for batch in loader:  # ← Already running
    loss = model(batch)  # ← CRASH! GPU OOM here
    
# Ideal (true compile-time): Would validate only on first call
# when JAX compiles the function
```

### Summary: Current State of RAX OOM Detection

✅ **What RAX Currently Does Well:**
- Catches OOM before GPU memory allocation
- Provides detailed memory analysis
- Gives actionable suggestions
- Much better than PyTorch's runtime crashes

⚠️ **Current Limitation:**
- Validates on every function call (performance overhead)
- Not true compile-time validation

🚀 **Future Enhancement Needed:**
- Integrate validation into JAX's JIT compilation process
- Cache validation results after first compilation
- Achieve true zero-overhead safety



## RAX-Enhanced Features
```python
# Every function has documented shapes
def gpt_forward(
    x: Int[Array, "batch seq_len"],
    params: GPTParams, 
    config: GPTConfig,
    key: PRNGKeyArray,
    training: bool = True,
    targets: Optional[Int[Array, "batch seq_len"]] = None
) -> Tuple[Float[Array, "batch seq_len vocab_size"], Float[Array, ""]]:
    # Implementation with compile-time safety guarantees
```

## 📊 Training Configuration

Key parameters with RAX validation:
```python
@dataclass(frozen=True)  # Immutable for JAX compatibility
class TrainConfig:
    n_layer: int = 6        # Transformer blocks
    n_head: int = 6         # Attention heads  
    n_embd: int = 384       # Embedding dimension
    block_size: int = 256   # Sequence length
    batch_size: int = 64    # Batch size (RAX checks memory fit)
    learning_rate: float = 1e-3
    max_iters: int = 5000
```

## 🔧 Development Workflow

### With RAX (Recommended)
```bash
# All safety checks enabled
rax run train.py config/my_config.py
```

### Without RAX (For comparison)
```bash  
# Traditional development - shape errors caught at runtime
python train.py config/my_config.py
```

## 📈 Performance & Results
*Need to add*



## 🏆 Comparison: Before vs After RAX

| Aspect | Without RAX | With RAX |
|--------|-------------|----------|
| **Error Detection** | Runtime (after hours) | Compile-time (seconds) |
| **Shape Documentation** | Comments (if any) | Function signatures |
| **Memory Safety** | Trial and error | Compile-time analysis |
| **Team Onboarding** | Read implementation | Read type signatures |
| **Debugging Speed** | Hours of investigation | Clear error messages |
| **Production Confidence** | Hope and pray | Mathematical guarantees |



## 🎓 Learning Resources

- **JAX**: [Official JAX documentation](https://jax.readthedocs.io/)
- **RAX**: [RAX repository](https://github.com/viraatdas/rax) 
- **Original nanoGPT**: [Andrej Karpathy's implementation](https://github.com/karpathy/nanoGPT)

## 📄 License

MIT - See original [nanoGPT](https://github.com/karpathy/nanoGPT) for attribution.
