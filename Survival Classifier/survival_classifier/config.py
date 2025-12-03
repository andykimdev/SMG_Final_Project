"""
Configuration file for Graph Transformer Survival Predictor 
Contains all hyperparameters (no hardcoded data paths).
"""

# Random seed for reproducibility
RANDOM_SEED = 42

# Data split ratios
TRAIN_RATIO = 0.80  # Changed from 0.85 for larger test set
VAL_RATIO = 0.10
TEST_RATIO = 0.10   # Changed from 0.05 for more reliable evaluation

# Model hyperparameters
MODEL = {
    'embedding_dim': 160,       # Dimension of token embeddings (v5: optimal size)
    'n_layers': 5,              # Number of transformer layers (v5: optimal depth)
    'n_heads': 8,               # Number of attention heads (v5: optimal)
    'ffn_dim': 640,             # Feedforward network dimension (4x embedding_dim)
    'dropout': 0.45,            # Dropout probability (v5: strong regularization)
    'pe_dim': 16,               # Graph positional encoding dimension (k eigenvectors)
    'graph_bias_scale': 0.25,   # Initial scale for graph-aware attention bias (v5: optimal balance)
}

# Training hyperparameters
TRAINING = {
    'batch_size': 64,           # Batch size for training
    'learning_rate': 3e-4,      # Learning rate for AdamW (v5: optimal LR)
    'weight_decay': 5e-4,       # L2 regularization (v5: strong regularization)
    'max_epochs': 50,           # Maximum number of training epochs
    'patience': 10,             # Early stopping patience
    'grad_clip': 1.0,           # Gradient clipping threshold
    'use_scheduler': True,      # Enable learning rate scheduling
    'scheduler_type': 'reduce_on_plateau',  # Reduce LR when validation metric plateaus
    'scheduler_factor': 0.5,    # Factor to reduce LR (multiply by 0.5)
    'scheduler_patience': 5,    # Wait 5 epochs before reducing LR (v5: optimal patience)
    'scheduler_min_lr': 1e-6,   # Minimum learning rate
}

# Graph prior processing
GRAPH_PRIOR = {
    'diffusion_beta': 0.5,      # Beta parameter for diffusion kernel
    'laplacian_type': 'normalized',  # 'normalized' or 'symmetric'
}

# Data preprocessing
DATA = {
    'missing_threshold': 0.5,   # Drop samples with >50% missing protein values
    'min_samples_per_class': 10,  # Minimum samples per cancer type
    'min_events': 50
}

# Matrics
METRICS = {
    'primary': 'c_index',       # Primary metric for model selection
    'track_loss': True,         # Track Cox loss
    'compute_calibration': False,  # Compute calibration curves (slow)
    'risk_groups': 3,           # Number of risk groups for stratification (low/med/high)
}