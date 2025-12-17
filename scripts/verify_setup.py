"""Verify project setup and check for required files."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def check_structure():
    """Check if project structure is correct."""
    required_dirs = [
        "src",
        "interpretability",
        "notebooks",
        "data/processed_datasets",
        "data/priors",
        "pretrained",
    ]
    
    missing = []
    for dir_path in required_dirs:
        full_path = PROJECT_ROOT / dir_path
        if not full_path.exists():
            missing.append(dir_path)
    
    if missing:
        print("Missing directories:")
        for d in missing:
            print(f"  - {d}")
        return False
    
    return True

def check_files():
    """Check for required Python files."""
    required_files = [
        "src/config.py",
        "src/model.py",
        "src/graph_prior.py",
        "src/dataset.py",
        "interpretability/utils.py",
        "interpretability/shap_analysis.py",
        "interpretability/attention_analysis.py",
        "interpretability/pca_baseline.py",
        "notebooks/model_demo.ipynb",
        "requirements.txt",
        "README.md",
    ]
    
    missing = []
    for file_path in required_files:
        full_path = PROJECT_ROOT / file_path
        if not full_path.exists():
            missing.append(file_path)
    
    if missing:
        print("Missing files:")
        for f in missing:
            print(f"  - {f}")
        return False
    
    return True

def check_data():
    """Check if data files are present."""
    data_dir = PROJECT_ROOT / "data"
    csv_paths = [
        data_dir / "processed_datasets" / "tcga_pancan_rppa_compiled.csv",
        data_dir / "tcga_pancan_rppa_compiled.csv",
    ]
    prior_paths = [
        data_dir / "priors" / "tcga_string_prior.npz",
        data_dir / "tcga_string_prior.npz",
    ]
    
    csv_found = any(p.exists() for p in csv_paths)
    prior_found = any(p.exists() for p in prior_paths)
    
    print(f"Data files:")
    print(f"  CSV: {'✓' if csv_found else '✗'}")
    print(f"  Prior: {'✓' if prior_found else '✗'}")
    
    return csv_found and prior_found

def check_pretrained():
    """Check if pretrained model is present."""
    model_path = PROJECT_ROOT / "pretrained" / "best_model.pt"
    exists = model_path.exists()
    print(f"Pretrained model: {'✓' if exists else '✗'}")
    return exists

if __name__ == "__main__":
    print("=" * 70)
    print("Project Setup Verification")
    print("=" * 70)
    
    print("\n1. Checking structure...")
    struct_ok = check_structure()
    
    print("\n2. Checking files...")
    files_ok = check_files()
    
    print("\n3. Checking data...")
    data_ok = check_data()
    
    print("\n4. Checking pretrained model...")
    model_ok = check_pretrained()
    
    print("\n" + "=" * 70)
    if struct_ok and files_ok:
        print("✓ Project structure is correct")
        if data_ok:
            print("✓ Data files found - ready for training")
        else:
            print("⚠ Data files not found - will show architecture only")
        if model_ok:
            print("✓ Pretrained model found - ready for analysis")
        else:
            print("⚠ Pretrained model not found")
    else:
        print("✗ Project structure incomplete")
    print("=" * 70)

