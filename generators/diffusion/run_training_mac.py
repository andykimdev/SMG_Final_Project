#!/usr/bin/env python3
"""
Training script for M4 Mac.
Run with: python run_training_mac.py

Note: Epochs and patience are controlled by config.py
      Currently set for full training (300 epochs, patience=30)
"""

import os
import sys
from pathlib import Path
import subprocess

def print_header():
    """Print training header."""
    # Import config to show actual values
    import config
    
    print("=" * 80)
    print("Graph-Aware Protein Diffusion Model - Training")
    print("=" * 80)
    print()
    print("Configuration (from config.py):")
    print(f"  Dataset:      TCGA PanCan RPPA (7,523 samples, 198 proteins)")
    print(f"  Cancer types: 32")
    print(f"  Epochs:       {config.TRAINING['max_epochs']} (early stopping patience={config.TRAINING['patience']})")
    print(f"  Batch size:   {config.TRAINING['batch_size']}")
    print(f"  Architecture: d_model={config.MODEL['embedding_dim']}, layers={config.MODEL['n_layers']}, heads={config.MODEL['n_heads']}")
    print(f"  Device:       MPS (Metal Performance Shaders)")
    print()
    print("Biological Priors:")
    print(f"  Graph prior:  STRING PPI network (198 proteins, ~1200 edges)")
    print(f"  Context:      Cancer type, stage, age, sex, molecular scores, survival outcomes")
    print()
    print("Output:")
    print(f"  Checkpoints:  outputs/checkpoints/")
    print(f"  Plots:        outputs/plots/")
    print(f"  Logs:         outputs/logs_*.txt")
    print()
    print("=" * 80)
    print()

def check_files():
    """Check if required files exist."""
    base_dir = Path(__file__).parent
    
    # Check dataset
    dataset_path = base_dir / "../processed_datasets/tcga_pancan_rppa_compiled.csv"
    if not dataset_path.exists():
        print("❌ ERROR: Dataset not found at", dataset_path)
        return False
    print("✅ Dataset found")
    
    # Check STRING prior
    prior_path = base_dir / "../priors/tcga_string_prior.npz"
    if not prior_path.exists():
        print("❌ ERROR: STRING prior not found at", prior_path)
        return False
    print("✅ STRING PPI graph prior found")
    
    # Note: Classifier checkpoint no longer needed (transfer learning disabled)
    print("ℹ️  Training from scratch (transfer learning disabled due to dim mismatch)")
    
    return True
    
def main():
    """Main training function."""
    # Change to script directory
    os.chdir(Path(__file__).parent)
    
    # Print header
    print_header()
    
    # Check files
    if not check_files():
        sys.exit(1)
    
    print()
    print("Starting training...")
    print("=" * 80)
    print()
    
    # Build command - always use --no_transfer since it's disabled anyway
    cmd = [
        sys.executable,  # Use current Python interpreter
        "train_diffusion.py",
        "--csv_path", "../processed_datasets/tcga_pancan_rppa_compiled.csv",
        "--prior_path", "../priors/tcga_string_prior.npz",
        "--output_dir", "outputs",
        "--device", "mps",
        "--num_workers", "0",
        "--no_transfer",  # Transfer learning disabled (dim mismatch)
    ]
    
    # Run training
    try:
        result = subprocess.run(cmd)
        exit_code = result.returncode
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Error running training: {e}")
        sys.exit(1)
    
    # Print results
    print()
    print("=" * 80)
    
    if exit_code == 0:
        print("✅ Training completed successfully!")
        print()
        print("Results saved to:")
        print("  outputs/checkpoints/best_model.pt")
        print("  outputs/plots/training_curves.png")
        print("  outputs/logs_*.txt")
        print()
        print("To validate with classifier, run:")
        print("  python validate_with_classifier.py")
    else:
        print(f"❌ Training failed with exit code {exit_code}")
        print()
        print("Common issues:")
        print("  - MPS not available: Try with --device cpu")
        print("  - Out of memory: Reduce batch_size in config.py")
        print("  - Missing dependencies: pip install -r requirements.txt")
    
    print("=" * 80)
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
