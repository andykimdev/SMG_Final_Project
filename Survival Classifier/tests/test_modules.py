"""
Quick test script to verify all survival prediction modules work.
Tests imports, data loading, model creation, and forward pass.
"""

import sys
import torch
import numpy as np

print("Testing Graph Transformer Survival Predictor modules...")
print("=" * 80)

# Test 1: Import all modules
print("\n1. Testing imports...")
try:
    from survival_classifier import config
    print("  ✓ config imported")
except Exception as e:
    print(f"  ✗ Failed to import config: {e}")
    sys.exit(1)

try:
    from survival_classifier.data import load_graph_prior, get_graph_features_as_tensors
    print("  ✓ graph_prior imported")
except Exception as e:
    print(f"  ✗ Failed to import graph_prior: {e}")
    sys.exit(1)

try:
    from survival_classifier.data import load_and_preprocess_survival_data, create_survival_dataloaders
    print("  ✓ dataset imported")
except Exception as e:
    print(f"  ✗ Failed to import dataset: {e}")
    sys.exit(1)

try:
    from survival_classifier.models.graph_transformer import SurvivalGraphTransformer, CoxPHLoss, ConcordanceIndex
    print("  ✓ graph_transformer imported")
except Exception as e:
    print(f"  ✗ Failed to import graph_transformer: {e}")
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
    
    # Create model with dummy dimensions
    model = SurvivalGraphTransformer(
        n_proteins=graph_prior['A'].shape[0],
        n_clinical=10,  # Dummy number
        n_genomic=4,    # Dummy number
        diffusion_kernel=graph_tensors['K'],
        positional_encodings=graph_tensors['PE'],
        use_clinical=True,
        use_genomic=True,
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
    
    # Create dummy batch (dictionary format for multimodal)
    dummy_batch = {
        'protein': torch.randn(batch_size, n_proteins),
        'clinical': torch.randn(batch_size, 10),
        'genomic': torch.randn(batch_size, 4),
    }
    
    # Forward pass
    with torch.no_grad():
        risk_scores = model(dummy_batch)
    
    assert risk_scores.shape == (batch_size, 1), f"Unexpected output shape: {risk_scores.shape}"
    print(f"  ✓ Forward pass successful")
    print(f"  ✓ Input protein shape: {dummy_batch['protein'].shape}")
    print(f"  ✓ Output risk scores shape: {risk_scores.shape}")
except Exception as e:
    print(f"  ✗ Failed forward pass: {e}")
    sys.exit(1)

# Test 5: Test loss computation
print("\n5. Testing Cox loss computation...")
try:
    cox_loss = CoxPHLoss()
    
    # Create dummy survival data
    dummy_time = torch.rand(batch_size) * 100  # Random times 0-100 months
    dummy_event = torch.randint(0, 2, (batch_size,)).float()  # Random events
    
    # Compute loss
    loss = cox_loss(risk_scores, dummy_time, dummy_event)
    
    print(f"  ✓ Cox loss computed: {loss.item():.4f}")
    print(f"  ✓ Loss requires grad: {loss.requires_grad}")
except Exception as e:
    print(f"  ✗ Failed loss computation: {e}")
    sys.exit(1)

# Test 6: Test C-index computation
print("\n6. Testing C-index computation...")
try:
    c_index_metric = ConcordanceIndex()
    
    # Compute C-index
    with torch.no_grad():
        c_index = c_index_metric(risk_scores, dummy_time, dummy_event)
    
    print(f"  ✓ C-index computed: {c_index.item():.4f}")
    assert 0 <= c_index.item() <= 1, "C-index out of valid range [0, 1]"
    print(f"  ✓ C-index in valid range [0, 1]")
except Exception as e:
    print(f"  ✗ Failed C-index computation: {e}")
    sys.exit(1)

# Test 7: Test data loading (if CSV is available)
print("\n7. Testing survival data loading...")
try:
    csv_path = "../processed_datasets/tcga_pancan_rppa_compiled.csv"
    
    data_splits, survival_info, preprocessing_info = load_and_preprocess_survival_data(
        csv_path,
        graph_prior['protein_cols'],
        use_clinical=True,
        use_genomic=True
    )
    
    print(f"  ✓ Data loaded successfully")
    print(f"  ✓ Total samples: {survival_info['total_samples']}")
    print(f"  ✓ Events (deaths): {survival_info['total_events']} ({survival_info['event_rate']*100:.1f}%)")
    print(f"  ✓ Censored: {survival_info['total_censored']}")
    print(f"  ✓ Train samples: {len(data_splits['train'][3])}")
    print(f"  ✓ Val samples: {len(data_splits['val'][3])}")
    print(f"  ✓ Test samples: {len(data_splits['test'][3])}")
    
    # Test dataloader creation
    train_loader, val_loader, test_loader = create_survival_dataloaders(
        data_splits, batch_size=32
    )
    print(f"  ✓ DataLoaders created")
    
    # Test one batch
    batch = next(iter(train_loader))
    print(f"  ✓ Sample batch shapes:")
    print(f"      Protein: {batch['protein'].shape}")
    print(f"      Clinical: {batch['clinical'].shape}")
    print(f"      Genomic: {batch['genomic'].shape}")
    print(f"      Time: {batch['time'].shape}")
    print(f"      Event: {batch['event'].shape}")
    
    # Create model with real dimensions
    print("\n  Testing with real data dimensions...")
    real_model = SurvivalGraphTransformer(
        n_proteins=preprocessing_info['feature_dims']['protein'],
        n_clinical=preprocessing_info['feature_dims']['clinical'],
        n_genomic=preprocessing_info['feature_dims']['genomic'],
        diffusion_kernel=graph_tensors['K'],
        positional_encodings=graph_tensors['PE'],
        use_clinical=True,
        use_genomic=True,
    ).to(device)
    
    # Test forward pass with real batch
    with torch.no_grad():
        real_risk_scores = real_model(batch)
    
    print(f"  ✓ Forward pass with real data successful")
    print(f"  ✓ Risk scores shape: {real_risk_scores.shape}")
    
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
print("    --output_dir outputs/survival \\")
print("    --device cpu")