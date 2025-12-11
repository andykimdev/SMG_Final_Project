# Protein Expression Diffusion Model

Generative model for RPPA protein expression profiles using denoising diffusion with graph-aware transformers.

## Training

```bash
python train_diffusion.py \
    --csv_path ../processed_datasets/tcga_pancan_rppa_compiled.csv \
    --prior_path ../priors/tcga_string_prior.npz \
    --output_dir outputs \
    --device cuda
```

## Generation

```bash
python sample_and_evaluate.py \
    --checkpoint outputs/checkpoints/best_model.pt \
    --num_samples 1000 \
    --output_dir outputs/evaluation
```

## Architecture

- Hierarchical patient context encoding (cancer type, stage, demographics, molecular, survival)
- Graph-aware transformer with STRING PPI attention bias
- FiLM conditioning for timestep
- DDPM noise prediction

## Configuration

See `config.py` for all settings.
