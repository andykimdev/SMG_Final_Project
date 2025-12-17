"""Configuration for Linear Baseline Classifiers."""

RANDOM_SEED = 33

TRAIN_RATIO = 0.85
VAL_RATIO = 0.10
TEST_RATIO = 0.05

GRAPH_PRIOR = {
    'diffusion_beta': 0.5,
    'laplacian_type': 'normalized',
    'pe_dim': 16,  # Positional encoding dimension (for graph prior computation)
}

DATA = {
    'missing_threshold': 0.5,
    'min_samples_per_class': 10,
}
