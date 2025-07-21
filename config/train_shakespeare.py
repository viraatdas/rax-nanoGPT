"""
Configuration for training GPT on Shakespeare dataset
Small model that should converge quickly for testing
"""

# Data
dataset = 'shakespeare'

# Model architecture - small model for quick training
n_layer = 4
n_head = 4
n_embd = 128
block_size = 256
bias = False
dropout = 0.0

# Training
batch_size = 12
learning_rate = 6e-4
max_iters = 2000
warmup_iters = 100
lr_decay_iters = 2000
min_lr = 6e-5
weight_decay = 1e-1
grad_clip = 1.0

# Evaluation
eval_interval = 100
eval_iters = 200

# Checkpointing
out_dir = 'out-shakespeare'
checkpoint_interval = 500

# System
seed = 1337

# Logging
log_interval = 10

# Initialize from
init_from = 'scratch' 