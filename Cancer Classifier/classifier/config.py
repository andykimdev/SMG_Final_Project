"""
Configuration file for Graph Transformer Cancer Type Classifier.
Contains all hyperparameters (no hardcoded data paths).
"""

# Random seed for reproducibility
RANDOM_SEED = 42

# Data split ratios
TRAIN_RATIO = 0.85
VAL_RATIO = 0.10
TEST_RATIO = 0.05

# Model hyperparameters
MODEL = {
    'embedding_dim': 128,       # Dimension of token embeddings
    'n_layers': 4,              # Number of transformer layers
    'n_heads': 8,               # Number of attention heads
    'ffn_dim': 512,             # Feedforward network dimension
    'dropout': 0.1,             # Dropout probability
    'pe_dim': 16,               # Graph positional encoding dimension (k eigenvectors)
    'graph_bias_scale': 1.0,    # Initial scale for graph-aware attention bias
}

# Training hyperparameters
TRAINING = {
    'batch_size': 64,           # Batch size for training
    'learning_rate': 1e-4,      # Learning rate for AdamW
    'weight_decay': 1e-5,       # L2 regularization
    'max_epochs': 100,          # Maximum number of training epochs
    'patience': 15,             # Early stopping patience (epochs)
    'grad_clip': 1.0,           # Gradient clipping threshold
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
}
