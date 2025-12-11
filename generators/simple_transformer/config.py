import torch


DATA = {
    "csv_path": "processed_datasets/tcga_pancan_rppa_compiled.csv",
    "prior_path": "priors/tcga_string_prior.npz",
    # Keep only cancers with >=250 samples (from dataset stats)
    "selected_cancers": [
        "BRCA", "BLCA", "COAD", "KIRC", "LGG", "LUAD", "LUSC",
        "OV", "PRAD", "SKCM", "STAD", "THCA", "UCEC"
    ],
    "stage_buckets": ["STAGE I", "STAGE II", "STAGE III", "STAGE IV", "UNKNOWN"],
    "sex_categories": ["FEMALE", "MALE", "UNKNOWN"],
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "seed": 42,
}

MODEL = {
    "d_model": 128,
    "n_layers": 4,
    "n_heads": 4,
    "ffn_dim": 512,
    "dropout": 0.1,
    "context_dim": 160,  # total context embedding width
    "context_splits": {
        "cancer": 64,
        "stage": 32,
        "sex": 32,
        "age": 32,
    },
    "noise_dropout": 0.0,
    "activation": "gelu",
}

TRAINING = {
    "device": "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
    "batch_size": 64,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "max_epochs": 120,
    "patience": 20,
    "min_delta": 1e-4,
    "log_every": 50,
    "save_dir": "outputs",
}

SAMPLING = {
    "samples_per_class": 50,
    "output_dir": "outputs/generated",
}

