"""Configuration for Graph Transformer Classifier."""

RANDOM_SEED = 33

TRAIN_RATIO = 0.85
VAL_RATIO = 0.10
TEST_RATIO = 0.05

MODEL = {
    'embedding_dim': 128,
    'n_layers': 4,
    'n_heads': 8,
    'ffn_dim': 512,
    'dropout': 0.1,
    'pe_dim': 16,
    'graph_bias_scale': 1.0,
}

TRAINING = {
    'batch_size': 64,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'max_epochs': 100,
    'patience': 15,
    'grad_clip': 1.0,
}

GRAPH_PRIOR = {
    'diffusion_beta': 0.5,
    'laplacian_type': 'normalized',
}

DATA = {
    'missing_threshold': 0.5,
    'min_samples_per_class': 10,
}
