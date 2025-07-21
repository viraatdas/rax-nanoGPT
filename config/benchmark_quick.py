# Quick benchmark configuration for performance testing
out_dir = 'out-benchmark'
eval_interval = 50
eval_iters = 10
log_interval = 1

# Small model for quick benchmarking
batch_size = 32
block_size = 128
n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.1

# Quick training settings
max_iters = 100
warmup_iters = 10
learning_rate = 1e-3
min_lr = 1e-4

# Fixed seed for reproducibility
seed = 42
checkpoint_interval = 200  # Don't save checkpoints during benchmark 