"""
Quick test to verify RAX-nanoGPT training works correctly.
This runs just a few iterations to ensure everything is set up properly.
"""

import os
import sys
import subprocess


def test_training_with_rax() -> None:
    """Test training with RAX validation"""
    print("=" * 70)
    print("Testing RAX-nanoGPT Training")
    print("=" * 70)
    
    # Create a minimal config for quick testing
    test_config = """
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
"""
    
    # Write test config
    config_path = "config/test_quick.py"
    with open(config_path, 'w') as f:
        f.write(test_config)
    
    print(f"Created test config: {config_path}")
    print("\nRunning training with RAX validation...")
    print("-" * 70)
    
    # Run with RAX
    cmd = ["rax", "run", "train.py", config_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Training completed successfully with RAX!")
        print("\nOutput excerpt:")
        lines = result.stdout.split('\n')
        # Show first few and last few lines
        for line in lines[:10]:
            print(f"  {line}")
        if len(lines) > 20:
            print("  ...")
            for line in lines[-10:]:
                print(f"  {line}")
    else:
        print(f"❌ Training failed with return code: {result.returncode}")
        print(f"Error output:\n{result.stderr}")
    
    # Clean up test config
    os.remove(config_path)
    print(f"\nCleaned up test config: {config_path}")


def main() -> None:
    """Run the training test"""
    print("\n🧪 RAX-nanoGPT Training Test\n")
    
    # Check if we're in the right directory
    if not os.path.exists("train.py"):
        print("❌ Error: Please run this script from the rax-nanogpt root directory")
        sys.exit(1)
    
    # Check if data is prepared
    if not os.path.exists("data/shakespeare/train.bin"):
        print("❌ Error: Shakespeare data not prepared. Run:")
        print("   python data/shakespeare/prepare.py")
        sys.exit(1)
    
    test_training_with_rax()
    
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print("If the test passed, you can run full training with:")
    print("  rax run train.py config/train_shakespeare.py")
    print("\nRAX ensures:")
    print("  - All tensor shapes are validated before training")
    print("  - Memory requirements are checked")
    print("  - Type safety throughout the model")


if __name__ == "__main__":
    main() 