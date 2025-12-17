# Pretrained Model Weights

This directory contains pretrained model checkpoints for immediate use without training.

## Files

### `best_model.pt`
PyTorch checkpoint containing:
- `model_state_dict`: Trained model weights
- `config`: Model and training configuration
- `label_info`: Class names and metadata

**File size**: ~2-5 MB (depending on model size)

### `model_config.json`
JSON file with model hyperparameters for reference.

## Usage

The model will be automatically loaded if present in this directory. The notebook and scripts check for `pretrained/best_model.pt` first.

## Download Instructions

If pretrained weights are hosted externally:

1. Download `best_model.pt` to this directory
2. Ensure file is named exactly `best_model.pt`
3. Run the demo notebook - it will automatically detect and load the model

## Model Details

- **Architecture**: Graph Transformer with 4 layers, 8 attention heads
- **Training**: 50 epochs on TCGA RPPA data, 32 cancer types
- **Performance**: ~88% test accuracy, 87% macro F1 score

## Notes

- Model requires matching PPI prior file (see `data/README.md`)
- Checkpoint is compatible with PyTorch >= 2.0.0
- Model can be loaded on CPU or GPU

