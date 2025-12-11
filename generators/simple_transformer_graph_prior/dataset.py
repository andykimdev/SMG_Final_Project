"""
Re-export the baseline simple generator data pipeline so both models
train/evaluate on identical splits and preprocessing.
"""

from simple_generator.dataset import create_dataloaders, load_data

__all__ = ["create_dataloaders", "load_data"]

