"""Configuration for Graph-Aware Protein Diffusion Model."""

RANDOM_SEED = 42

TRAIN_RATIO = 0.85
VAL_RATIO = 0.10
TEST_RATIO = 0.05

DATA = {
    'train_ratio': 0.85,
    'val_ratio': 0.10,
    'test_ratio': 0.05,
    'missing_threshold': 0.5,
    'min_samples_per_class': 10,
    
    'context_features': {
        'categorical': [
            'CANCER_TYPE_ACRONYM',
            'SEX',
            'AJCC_PATHOLOGIC_TUMOR_STAGE',
        ],
        'continuous': [
            'AGE',
            'ANEUPLOIDY_SCORE',
            'TMB_NONSYNONYMOUS',
            'MSI_SCORE_MANTIS',
            'TBL_SCORE',
        ],
        'survival': [
            ('OS_STATUS', 'OS_MONTHS'),
            ('PFS_STATUS', 'PFS_MONTHS'),
            ('DSS_STATUS', 'DSS_MONTHS'),
            ('DFS_STATUS', 'DFS_MONTHS'),
        ],
    },
    
    'num_cancer_types': 32,
    'num_sexes': 3,
    'num_stages': 20,
    'num_survival_status': 3,
}

DIFFUSION = {
    'timesteps': 100,
    'schedule': 'linear',
    'cosine_s': 0.008,
    'beta_start': 1e-4,
    'beta_end': 0.02,
    
    'sampling': {
        'num_samples': 16,
        'ddim_steps': 50,
        'eta': 0.0,
    },
    
    'variance_type': 'fixed_small',
}

GRAPH_PRIOR = {
    'diffusion_beta': 0.5,
    'laplacian_type': 'normalized',
}

MODEL = {
    'embedding_dim': 256,
    'n_layers': 5,
    'n_heads': 8,
    'ffn_dim': 896,
    'dropout': 0.1,
    'pe_dim': 16,
    'graph_bias_scale': 1.0,
    
    'context_encoder': {
        'cancer_type_dim': 64,
        'stage_dim': 32,
        'age_dim': 16,
        'sex_dim': 8,
        'molecular_dim': 32,
        'survival_dim': 64,
        'survival_per_outcome_dim': 16,
        'context_dim': 256,
        'fusion_layers': 3,
    },
    
    'time_embedding': {
        'method': 'sinusoidal',
        'dim': 256,
    },
    
    'conditioning': {
        'use_ctx_token': True,
        'use_film': False,
        'use_cross_attention': False,
    },
    
    'output_head': {
        'hidden_dim': 128,
        'activation': 'gelu',
    },
}

TRAINING = {
    'batch_size': 64,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'max_epochs': 100,
    'warmup_epochs': 10,
    'grad_clip': 1.0,
    'patience': 15,
    'min_delta': 1e-5,
    'loss_type': 'mae',
    'huber_delta': 0.1,
    'protein_weights': None,
    'use_ema': False,
    'ema_decay': 0.9999,
    
    'scheduler': {
        'type': 'reduce_on_plateau',
        'factor': 0.5,
        'patience': 10,
        'min_lr': 1e-6,
    },
    
    'log_every': 50,
    'eval_every': 1,
    'save_every': 10,
}

TRANSFER_LEARNING = {
    'use_pretrained': False,
    'classifier_checkpoint': '../Cancer Classifier/outputs/checkpoints/best_model.pt',
    'freeze_graph_embeddings': False,
    'freeze_transformer': False,
    'lr_multiplier_embeddings': 0.1,
    'lr_multiplier_transformer': 0.5,
    'lr_multiplier_new': 1.0,
}

EVALUATION = {
    'num_eval_samples': 1000,
    'metrics': [
        'mse',
        'protein_wise_ks',
        'mmd',
        'pathway_coherence',
        'ppi_consistency',
        'classifier_accuracy',
    ],
    'pathway_analysis': {
        'min_pathway_size': 5,
        'use_string_modules': True,
    },
    'visualizations': [
        'tsne',
        'umap',
        'protein_distributions',
        'correlation_matrices',
    ],
}

COMPUTE = {
    'device': 'cpu',
    'num_workers': 0,
    'pin_memory': True,
    'mixed_precision': False,
}

PATHS = {
    'csv_path': '../processed_datasets/tcga_pancan_rppa_compiled.csv',
    'prior_path': '../priors/tcga_string_prior.npz',
    'output_dir': 'outputs',
    'checkpoint_dir': 'outputs/checkpoints',
    'samples_dir': 'outputs/samples',
    'plots_dir': 'outputs/plots',
    'results_dir': 'outputs/results',
}

EXPERIMENTAL = {
    'classifier_free_guidance': {
        'enabled': False,
        'dropout_prob': 0.1,
        'guidance_scale': 1.0,
    },
    'self_conditioning': {
        'enabled': False,
        'prob': 0.5,
    },
    'protein_masking': {
        'enabled': False,
        'mask_prob': 0.15,
    },
}


def set_seed(seed=RANDOM_SEED):
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
