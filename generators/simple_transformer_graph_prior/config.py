"""
Configuration for the graph-prior-aware simple transformer.
Inherits most defaults from the baseline simple_generator config.
"""

import copy
import torch

from simple_generator import config as base_config

# Reuse the same data setup to guarantee identical splits and feature order.
DATA = copy.deepcopy(base_config.DATA)

# Training hyperparameters largely mirror the baseline; only the save directory differs.
TRAINING = copy.deepcopy(base_config.TRAINING)
TRAINING["save_dir"] = "outputs_graph_prior"

# Transformer architecture stays the same but adds a few graph-specific knobs.
MODEL = copy.deepcopy(base_config.MODEL)
MODEL.update(
    {
        "graph_pe_dim": 32,
        "graph_feature_dropout": 0.05,
        "diffusion_blend_init": 0.0,  # sigmoid -> 0.5 initial blend
    }
)

# Graph prior processing controls.
GRAPH = {
    "prior_path": DATA["prior_path"],
    "laplacian_type": "symmetric",
    "diffusion_beta": 0.5,
}

# Logging/plotting helpers.
LOGGING = {
    "history_file": f"{TRAINING['save_dir']}/training_history.json",
}

# Keep sampling defaults identical so downstream utilities continue to work.
SAMPLING = copy.deepcopy(base_config.SAMPLING)

