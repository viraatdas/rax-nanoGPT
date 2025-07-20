"""
Full definition of a GPT Language Model in JAX with RAX type annotations.
Based on the nanoGPT implementation by Andrej Karpathy.
"""

import math
from dataclasses import dataclass
from typing import Tuple, Optional, NamedTuple

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float, Int, Bool, PRNGKeyArray
import numpy as np


@dataclass(frozen=True)
class GPTConfig:
    """Configuration for GPT model"""
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded to nearest multiple of 64
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2


class GPTParams(NamedTuple):
    """Parameters for the GPT model"""
    # Token embeddings
    wte: Float[Array, "vocab_size n_embd"]
    # Position embeddings
    wpe: Float[Array, "block_size n_embd"]
    # Layer parameters
    blocks: list  # List of block parameters
    # Final layer norm
    ln_f_weight: Float[Array, "n_embd"]
    ln_f_bias: Optional[Float[Array, "n_embd"]]
    # Language model head (shares weight with wte)


class BlockParams(NamedTuple):
    """Parameters for a transformer block"""
    # Layer norm 1
    ln_1_weight: Float[Array, "n_embd"]
    ln_1_bias: Optional[Float[Array, "n_embd"]]
    # Attention
    c_attn_weight: Float[Array, "n_embd 3*n_embd"]
    c_attn_bias: Optional[Float[Array, "3*n_embd"]]
    c_proj_weight: Float[Array, "n_embd n_embd"]
    c_proj_bias: Optional[Float[Array, "n_embd"]]
    # Layer norm 2
    ln_2_weight: Float[Array, "n_embd"]
    ln_2_bias: Optional[Float[Array, "n_embd"]]
    # MLP
    c_fc_weight: Float[Array, "n_embd 4*n_embd"]
    c_fc_bias: Optional[Float[Array, "4*n_embd"]]
    c_proj_2_weight: Float[Array, "4*n_embd n_embd"]
    c_proj_2_bias: Optional[Float[Array, "n_embd"]]


def layer_norm(
    x: Float[Array, "... n_embd"], 
    weight: Float[Array, "n_embd"], 
    bias: Optional[Float[Array, "n_embd"]] = None,
    eps: float = 1e-5
) -> Float[Array, "... n_embd"]:
    """Layer normalization"""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    x = (x - mean) / jnp.sqrt(var + eps)
    x = x * weight
    if bias is not None:
        x = x + bias
    return x


def gelu(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """GELU activation function"""
    return 0.5 * x * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * x**3)))


def dropout(
    x: Float[Array, "..."], 
    key: Optional[PRNGKeyArray],
    rate: float,
    training: bool = True
) -> Float[Array, "..."]:
    """Dropout layer"""
    if not training or rate == 0.0 or key is None:
        return x
    keep_prob = 1 - rate
    mask = random.bernoulli(key, keep_prob, x.shape)
    return x * mask / keep_prob


def causal_self_attention(
    x: Float[Array, "batch seq n_embd"],
    params: BlockParams,
    config: GPTConfig,
    key: Optional[PRNGKeyArray] = None,
    training: bool = True
) -> Float[Array, "batch seq n_embd"]:
    """Multi-head causal self attention"""
    batch, seq_len, n_embd = x.shape
    n_head = config.n_head
    
    # Linear projection to q, k, v
    qkv = jnp.dot(x, params.c_attn_weight)
    if params.c_attn_bias is not None:
        qkv = qkv + params.c_attn_bias
    
    # Split into q, k, v
    q, k, v = jnp.split(qkv, 3, axis=-1)
    
    # Reshape for multi-head attention
    head_dim = n_embd // n_head
    q = q.reshape(batch, seq_len, n_head, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(batch, seq_len, n_head, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(batch, seq_len, n_head, head_dim).transpose(0, 2, 1, 3)
    
    # Compute attention scores
    att = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(head_dim)
    
    # Apply causal mask
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    mask = mask.reshape(1, 1, seq_len, seq_len)
    att = jnp.where(mask == 0, -1e10, att)
    
    # Softmax
    att = jax.nn.softmax(att, axis=-1)
    
    # Dropout on attention weights
    if key is not None:
        key, subkey = random.split(key)
        att = dropout(att, subkey, config.dropout, training)
    
    # Apply attention to values
    y = jnp.matmul(att, v)
    
    # Reshape back
    y = y.transpose(0, 2, 1, 3).reshape(batch, seq_len, n_embd)
    
    # Output projection
    y = jnp.dot(y, params.c_proj_weight)
    if params.c_proj_bias is not None:
        y = y + params.c_proj_bias
    
    # Residual dropout
    if key is not None:
        _, subkey = random.split(key)
        y = dropout(y, subkey, config.dropout, training)
    
    return y


def mlp(
    x: Float[Array, "batch seq n_embd"],
    params: BlockParams,
    config: GPTConfig,
    key: Optional[PRNGKeyArray] = None,
    training: bool = True
) -> Float[Array, "batch seq n_embd"]:
    """MLP block"""
    # First linear
    x = jnp.dot(x, params.c_fc_weight)
    if params.c_fc_bias is not None:
        x = x + params.c_fc_bias
    
    # GELU activation
    x = gelu(x)
    
    # Second linear
    x = jnp.dot(x, params.c_proj_2_weight)
    if params.c_proj_2_bias is not None:
        x = x + params.c_proj_2_bias
    
    # Dropout
    if key is not None:
        x = dropout(x, key, config.dropout, training)
    
    return x


def transformer_block(
    x: Float[Array, "batch seq n_embd"],
    params: BlockParams,
    config: GPTConfig,
    key: Optional[PRNGKeyArray] = None,
    training: bool = True
) -> Float[Array, "batch seq n_embd"]:
    """Transformer block"""
    # Split key for attention and mlp
    if key is not None:
        key1, key2 = random.split(key)
    else:
        key1 = key2 = None
    
    # Self attention with residual
    ln1_out = layer_norm(x, params.ln_1_weight, params.ln_1_bias)
    attn_out = causal_self_attention(ln1_out, params, config, key1, training)
    x = x + attn_out
    
    # MLP with residual
    ln2_out = layer_norm(x, params.ln_2_weight, params.ln_2_bias)
    mlp_out = mlp(ln2_out, params, config, key2, training)
    x = x + mlp_out
    
    return x


def gpt_forward(
    idx: Int[Array, "batch seq"],
    params: GPTParams,
    config: GPTConfig,
    key: Optional[PRNGKeyArray] = None,
    training: bool = True,
    targets: Optional[Int[Array, "batch seq"]] = None
) -> Tuple[Float[Array, "batch seq vocab_size"], Optional[Float[Array, ""]]]:
    """Forward pass of GPT model"""
    batch, seq_len = idx.shape
    assert seq_len <= config.block_size, f"Sequence length {seq_len} exceeds block size {config.block_size}"
    
    # Token embeddings
    tok_emb = params.wte[idx]  # (batch, seq, n_embd)
    
    # Position embeddings
    pos = jnp.arange(seq_len)
    pos_emb = params.wpe[pos]  # (seq, n_embd)
    
    # Combine embeddings
    x = tok_emb + pos_emb
    
    # Dropout
    if key is not None:
        keys = random.split(key, config.n_layer + 1)
        x = dropout(x, keys[0], config.dropout, training)
    else:
        keys = [None] * (config.n_layer + 1)
    
    # Apply transformer blocks
    for i, block_params in enumerate(params.blocks):
        x = transformer_block(x, block_params, config, keys[i+1], training)
    
    # Final layer norm
    x = layer_norm(x, params.ln_f_weight, params.ln_f_bias)
    
    # Language model head (using weight tying with wte)
    logits = jnp.dot(x, params.wte.T)  # (batch, seq, vocab_size)
    
    # Calculate loss if targets provided
    loss = None
    if targets is not None:
        # Flatten logits and targets
        logits_flat = logits.reshape(-1, config.vocab_size)
        targets_flat = targets.reshape(-1)
        
        # Cross entropy loss
        log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
        loss = -jnp.mean(log_probs[jnp.arange(targets_flat.shape[0]), targets_flat])
    
    return logits, loss


def init_gpt_params(config: GPTConfig, key: PRNGKeyArray) -> GPTParams:
    """Initialize GPT parameters"""
    keys = random.split(key, 2 + config.n_layer * 10)
    key_idx = 0
    
    # Token and position embeddings
    wte = random.normal(keys[key_idx], (config.vocab_size, config.n_embd)) * 0.02
    key_idx += 1
    wpe = random.normal(keys[key_idx], (config.block_size, config.n_embd)) * 0.02
    key_idx += 1
    
    # Initialize blocks
    blocks = []
    for _ in range(config.n_layer):
        block_params = BlockParams(
            # Layer norm 1
            ln_1_weight=jnp.ones(config.n_embd),
            ln_1_bias=jnp.zeros(config.n_embd) if config.bias else None,
            # Attention
            c_attn_weight=random.normal(keys[key_idx], (config.n_embd, 3 * config.n_embd)) * 0.02,
            c_attn_bias=jnp.zeros(3 * config.n_embd) if config.bias else None,
            c_proj_weight=random.normal(keys[key_idx+1], (config.n_embd, config.n_embd)) * (0.02 / math.sqrt(2 * config.n_layer)),
            c_proj_bias=jnp.zeros(config.n_embd) if config.bias else None,
            # Layer norm 2
            ln_2_weight=jnp.ones(config.n_embd),
            ln_2_bias=jnp.zeros(config.n_embd) if config.bias else None,
            # MLP
            c_fc_weight=random.normal(keys[key_idx+2], (config.n_embd, 4 * config.n_embd)) * 0.02,
            c_fc_bias=jnp.zeros(4 * config.n_embd) if config.bias else None,
            c_proj_2_weight=random.normal(keys[key_idx+3], (4 * config.n_embd, config.n_embd)) * (0.02 / math.sqrt(2 * config.n_layer)),
            c_proj_2_bias=jnp.zeros(config.n_embd) if config.bias else None,
        )
        blocks.append(block_params)
        key_idx += 4
    
    # Final layer norm
    ln_f_weight = jnp.ones(config.n_embd)
    ln_f_bias = jnp.zeros(config.n_embd) if config.bias else None
    
    params = GPTParams(
        wte=wte,
        wpe=wpe,
        blocks=blocks,
        ln_f_weight=ln_f_weight,
        ln_f_bias=ln_f_bias
    )
    
    # Count parameters
    def count_params(params):
        if isinstance(params, (list, tuple)):
            return sum(count_params(p) for p in params)
        elif hasattr(params, '_fields'):
            return sum(count_params(getattr(params, field)) for field in params._fields)
        elif isinstance(params, jnp.ndarray):
            return params.size
        elif params is None:
            return 0
        else:
            return 0
    
    num_params = count_params(params)
    print(f"Number of parameters: {num_params/1e6:.2f}M")
    
    return params 