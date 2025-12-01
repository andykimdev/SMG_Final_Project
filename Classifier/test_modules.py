"""
Quick test script to verify all modules can be imported and basic functionality works.
"""

import sys
import torch
import numpy as np

print("Testing Graph Transformer Classifier modules...")
print("=" * 80)

# Test 1: Import all modules
print("\n1. Testing imports...")
try:
    import config
    print("  ✓ config.py imported")
except Exception as e:
    print(f"  ✗ Failed to import config.py: {e}")
    sys.exit(1)

try:
    from graph_prior import load_graph_prior, get_graph_features_as_tensors
    print("  ✓ graph_prior.py imported")
except Exception as e:
    print(f"  ✗ Failed to import graph_prior.py: {e}")
    sys.exit(1)

try:
    from dataset_tcga_rppa import load_and_preprocess_data, create_dataloaders
    print("  ✓ dataset_tcga_rppa.py imported")
except Exception as e:
    print(f"  ✗ Failed to import dataset_tcga_rppa.py: {e}")
    sys.exit(1)

try:
    from graph_transformer_classifier import GraphTransformerClassifier
    print("  ✓ graph_transformer_classifier.py imported")
except Exception as e:
    print(f"  ✗ Failed to import graph_transformer_classifier.py: {e}")
    sys.exit(1)

# Test 2: Load graph prior
print("\n2. Testing graph prior loading...")
try:
    prior_path = "../priors/tcga_string_prior.npz"
    graph_prior = load_graph_prior(prior_path)
    
    assert 'A' in graph_prior, "Missing adjacency matrix"
    assert 'K' in graph_prior, "Missing diffusion kernel"
    assert 'PE' in graph_prior, "Missing positional encodings"
    assert 'protein_cols' in graph_prior, "Missing protein columns"
    
    print(f"  ✓ Loaded prior with {len(graph_prior['protein_cols'])} proteins")
    print(f"  ✓ Adjacency matrix shape: {graph_prior['A'].shape}")
    print(f"  ✓ Diffusion kernel shape: {graph_prior['K'].shape}")
    print(f"  ✓ PE shape: {graph_prior['PE'].shape}")
except Exception as e:
    print(f"  ✗ Failed to load graph prior: {e}")
    sys.exit(1)

# Test 3: Create model
print("\n3. Testing model creation...")
try:
    device = 'cpu'
    graph_tensors = get_graph_features_as_tensors(graph_prior, device=device)
    
    model = GraphTransformerClassifier(
        n_proteins=graph_prior['A'].shape[0],
        n_classes=30,  # Dummy number
        diffusion_kernel=graph_tensors['K'],
        positional_encodings=graph_tensors['PE'],
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Model created with {n_params:,} parameters")
except Exception as e:
    print(f"  ✗ Failed to create model: {e}")
    sys.exit(1)

# Test 4: Test forward pass
print("\n4. Testing forward pass...")
try:
    batch_size = 8
    n_proteins = graph_prior['A'].shape[0]
    
    # Create dummy input
    x = torch.randn(batch_size, n_proteins)
    
    # Forward pass
    with torch.no_grad():
        logits = model(x)
    
    assert logits.shape == (batch_size, 30), f"Unexpected output shape: {logits.shape}"
    print(f"  ✓ Forward pass successful")
    print(f"  ✓ Input shape: {x.shape}")
    print(f"  ✓ Output shape: {logits.shape}")
except Exception as e:
    print(f"  ✗ Failed forward pass: {e}")
    sys.exit(1)

# Test 5: Test data loading (if CSV is available)
print("\n5. Testing data loading...")
try:
    csv_path = "../processed_datasets/tcga_pancan_rppa_compiled.csv"
    
    data_splits, label_info, scaler = load_and_preprocess_data(
        csv_path,
        graph_prior['protein_cols']
    )
    
    print(f"  ✓ Data loaded successfully")
    print(f"  ✓ Number of classes: {label_info['n_classes']}")
    print(f"  ✓ Train samples: {len(data_splits['train'][1])}")
    print(f"  ✓ Val samples: {len(data_splits['val'][1])}")
    print(f"  ✓ Test samples: {len(data_splits['test'][1])}")
    
    # Test dataloader creation
    train_loader, val_loader, test_loader = create_dataloaders(data_splits, batch_size=32)
    print(f"  ✓ DataLoaders created")
    
    # Test one batch
    x_batch, y_batch = next(iter(train_loader))
    print(f"  ✓ Sample batch shape: {x_batch.shape}, {y_batch.shape}")
    
except FileNotFoundError:
    print(f"  ⚠ CSV file not found at {csv_path}, skipping data loading test")
except Exception as e:
    print(f"  ✗ Failed to load data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✓ All tests passed!")
print("=" * 80)
print("\nYou can now run the training script:")
print("  python train_and_eval.py \\")
print("    --csv_path ../processed_datasets/tcga_pancan_rppa_compiled.csv \\")
print("    --prior_path ../priors/tcga_string_prior.npz \\")
print("    --output_dir outputs \\")
print("    --device cpu")

