# Proteomics VGAE

Variational Graph Autoencoder for human plasma proteomics data from the Human Protein Atlas using protein interactions from STRING.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Place your data files in `data/raw/`:
   - `protein_interactions.tsv` (STRING format)
   - `hpa_plasma_proteins.csv` (HPA format)

2. Edit `config.yaml` with your settings

3. Run training:
```bash
python scripts/run.py
```

This will output:
- `vgae_model.pth` - Trained model
- `latent_embeddings.npy` - Protein embeddings
- `training_history.png` - Training curves