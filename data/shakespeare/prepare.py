"""
Prepare the Shakespeare dataset for training.
Downloads the tiny Shakespeare dataset and tokenizes it using tiktoken (GPT-2 tokenizer).
"""

import os
import requests
import numpy as np
import tiktoken
from typing import Tuple


def download_shakespeare() -> str:
    """Download the tiny Shakespeare dataset"""
    # Download the tiny Shakespeare dataset
    input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
    if not os.path.exists(input_file_path):
        print("Downloading Shakespeare dataset...")
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(input_file_path, 'w') as f:
            f.write(requests.get(data_url).text)
        print(f"Dataset saved to {input_file_path}")
    else:
        print("Shakespeare dataset already exists")
    
    with open(input_file_path, 'r') as f:
        data = f.read()
    
    print(f"Dataset length: {len(data):,} characters")
    return data


def tokenize_and_save(data: str, train_ratio: float = 0.9) -> Tuple[np.ndarray, np.ndarray, tiktoken.Encoding]:
    """Tokenize the data using tiktoken and save train/val splits"""
    # Initialize the GPT-2 tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # Tokenize the entire dataset
    print("Tokenizing dataset...")
    tokens = enc.encode_ordinary(data)
    tokens = np.array(tokens, dtype=np.uint16)
    print(f"Total tokens: {len(tokens):,}")
    
    # Calculate split
    n = len(tokens)
    train_n = int(train_ratio * n)
    
    # Split the data
    train_tokens = tokens[:train_n]
    val_tokens = tokens[train_n:]
    
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens: {len(val_tokens):,}")
    
    # Save to binary files
    train_path = os.path.join(os.path.dirname(__file__), 'train.bin')
    val_path = os.path.join(os.path.dirname(__file__), 'val.bin')
    
    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)
    
    print(f"Saved train data to {train_path}")
    print(f"Saved validation data to {val_path}")
    
    # Save some metadata
    meta = {
        'vocab_size': enc.n_vocab,
        'tokenizer': 'gpt2',
        'train_tokens': len(train_tokens),
        'val_tokens': len(val_tokens)
    }
    
    import json
    meta_path = os.path.join(os.path.dirname(__file__), 'meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"Saved metadata to {meta_path}")
    
    return train_tokens, val_tokens, enc


def main() -> None:
    """Main function to prepare the Shakespeare dataset"""
    print("Preparing Shakespeare dataset for nanoGPT...")
    
    # Download the dataset
    data = download_shakespeare()
    
    # Show a sample
    print("\nFirst 200 characters:")
    print(data[:200])
    print("...\n")
    
    # Tokenize and save
    train_tokens, val_tokens, enc = tokenize_and_save(data)
    
    # Test loading
    print("\nTesting data loading...")
    train_path = os.path.join(os.path.dirname(__file__), 'train.bin')
    loaded_train = np.fromfile(train_path, dtype=np.uint16)
    assert len(loaded_train) == len(train_tokens)
    assert np.array_equal(loaded_train, train_tokens)
    print("✓ Train data loads correctly")
    
    val_path = os.path.join(os.path.dirname(__file__), 'val.bin')
    loaded_val = np.fromfile(val_path, dtype=np.uint16)
    assert len(loaded_val) == len(val_tokens)
    assert np.array_equal(loaded_val, val_tokens)
    print("✓ Validation data loads correctly")
    
    # Show a decoded sample
    print("\nSample decoded text from training set:")
    sample = train_tokens[:100]
    decoded = enc.decode(sample.tolist())
    print(decoded)
    
    print("\nDataset preparation complete!")


if __name__ == "__main__":
    main() 