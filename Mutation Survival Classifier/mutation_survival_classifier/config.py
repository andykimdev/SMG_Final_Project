"""
Configuration file for Mutation-Augmented Graph Transformer Survival Predictor
Uses RPPA protein expression + genomic mutations for survival prediction.
"""

# Random seed for reproducibility
RANDOM_SEED = 42

# Data split ratios
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# Model hyperparameters (based on v7_shallow_wide from Survival Classifier)
MODEL = {
    'embedding_dim': 256,       # Dimension of token embeddings
    'n_layers': 3,              # Number of transformer layers (shallow works best)
    'n_heads': 8,               # Number of attention heads
    'ffn_dim': 1024,            # Feedforward network dimension
    'dropout': 0.4,             # Dropout probability
    'pe_dim': 16,               # Graph positional encoding dimension
    'graph_bias_scale': 0.25,   # Initial scale for graph-aware attention bias
}

# Training hyperparameters
TRAINING = {
    'batch_size': 64,           # Batch size for training
    'learning_rate': 0.0003,    # Learning rate for AdamW
    'weight_decay': 0.0005,     # L2 regularization
    'max_epochs': 50,           # Maximum number of training epochs
    'patience': 10,             # Early stopping patience
    'grad_clip': 1.0,           # Gradient clipping threshold
    'use_scheduler': True,      # Enable learning rate scheduling
    'scheduler_type': 'reduce_on_plateau',
    'scheduler_factor': 0.5,
    'scheduler_patience': 5,
    'scheduler_min_lr': 1e-6,
}

# Graph prior processing
GRAPH_PRIOR = {
    'diffusion_beta': 0.5,           # Beta parameter for diffusion kernel
    'laplacian_type': 'normalized',  # 'normalized' or 'symmetric'
}

# Data preprocessing
DATA = {
    'missing_threshold': 0.5,     # Drop samples with >50% missing protein values
    'min_samples_per_class': 10,  # Minimum samples per cancer type
    'min_events': 50              # Minimum events for meaningful survival analysis
}

# Metrics
METRICS = {
    'primary': 'c_index',       # Primary metric for model selection
    'track_loss': True,         # Track Cox loss
    'compute_calibration': False,  # Compute calibration curves (slow)
    'risk_groups': 3,           # Number of risk groups for stratification
}
