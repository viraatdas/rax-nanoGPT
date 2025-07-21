
# Quick test configuration
from train import TrainConfig

config = TrainConfig(
    # Small model for quick testing
    n_layer=2,
    n_head=2,
    n_embd=64,
    block_size=128,
    
    # Minimal training
    batch_size=4,
    max_iters=10,  # Just 10 iterations
    eval_interval=5,
    eval_iters=2,
    
    # Output
    out_dir='out-test',
    checkpoint_interval=1000,  # Don't save checkpoints for test
    
    # Logging
    log_interval=1,
)
