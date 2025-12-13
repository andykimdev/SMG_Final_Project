# Model Architecture Verification

## 1. ProteinOnlyMLP ✅

**Correct!** Only uses protein features.

### Code Evidence:
```python
# blinded_survival_classifier/models/baseline.py:359-360
def forward(self, batch: dict) -> torch.Tensor:
    return self.model(batch['protein'])  # ONLY protein!
```

**Features used:**
- ✅ Protein expression (~200-300 features)
- ❌ Clinical (Age, Sex, Race, Ancestry) - EXCLUDED
- ❌ Genomic - EXCLUDED

**Purpose:** Tests if protein expression alone is sufficient for survival prediction.

---

## 2. Graph Transformer ✅

**Correct!** Uses STRING PPI network prior.

### Code Evidence:
```python
# scripts/train_and_eval.py:311-312
graph_prior = load_graph_prior(args.prior_path)
graph_tensors = get_graph_features_as_tensors(graph_prior, device=args.device)

# scripts/train_and_eval.py:344-352
model = SurvivalGraphTransformer(
    n_proteins=preprocessing_info['feature_dims']['protein'],
    n_clinical=preprocessing_info['feature_dims']['clinical'],
    n_genomic=preprocessing_info['feature_dims']['genomic'],
    diffusion_kernel=graph_tensors['K'],      # ✅ STRING diffusion kernel
    positional_encodings=graph_tensors['PE'],  # ✅ Graph positional encodings
    use_clinical=use_clinical,
    use_genomic=use_genomic,
)
```

**What the STRING prior provides:**
1. **Diffusion Kernel (K)**: Graph structure for attention bias
   - Built from STRING protein-protein interaction network
   - Encodes which proteins interact biologically
   - Used to bias attention towards connected proteins

2. **Positional Encodings (PE)**: Graph-based position embeddings
   - Computed from graph Laplacian eigenvectors
   - Gives each protein a unique "position" in the network
   - Helps model distinguish protein roles in pathways

**Features used:**
- ✅ Protein expression (~200-300 features) WITH graph structure
- ✅ Clinical (Age, Sex, Race, Ancestry) - 4 features
- ❌ Genomic - EXCLUDED (for blinded model)

**Purpose:** Tests if protein interaction network topology improves prediction.

---

## 3. Other Baselines

### MLPSurvivalModel
```python
# Line 57-64
def forward(self, batch: dict) -> torch.Tensor:
    features = torch.cat([
        batch['protein'],
        batch['clinical'],
        batch['genomic']  # Will be empty for blinded
    ], dim=1)
    return self.model(features)
```
**Features:** Protein + Clinical (no graph structure)

### VanillaTransformerSurvival
**Features:** Protein + Clinical (standard attention, no graph bias)

---

## Summary Comparison

| Model | Proteins | Clinical | Genomic | Graph Prior |
|-------|----------|----------|---------|-------------|
| **ProteinOnlyMLP** | ✅ | ❌ | ❌ | ❌ |
| **MLPSurvivalModel** | ✅ | ✅ | ❌ | ❌ |
| **VanillaTransformerSurvival** | ✅ | ✅ | ❌ | ❌ |
| **SurvivalGraphTransformer** | ✅ | ✅ | ❌ | **✅ STRING** |

## Key Insights

### ProteinOnlyMLP vs. MLPSurvivalModel
Compares: **Do clinical demographics (Age, Sex, Race) add value?**
- Same architecture (MLP)
- ProteinOnly: Just proteins
- MLP: Proteins + clinical

### MLPSurvivalModel vs. SurvivalGraphTransformer
Compares: **Does STRING PPI network topology add value?**
- Same features (Protein + Clinical)
- MLP: No structure
- GraphTransformer: Uses STRING network

### VanillaTransformer vs. SurvivalGraphTransformer
Compares: **Does graph-aware attention beat vanilla attention?**
- Same features (Protein + Clinical)
- Both use transformer architecture
- Vanilla: Standard self-attention
- Graph: STRING-biased attention

## Verification Checklist

- [x] ProteinOnlyMLP uses ONLY protein features
- [x] Graph Transformer receives STRING prior (K and PE)
- [x] Graph Transformer uses diffusion kernel for attention bias
- [x] Graph Transformer uses positional encodings
- [x] All blinded models exclude genomic features
- [x] All blinded models exclude cancer type/staging info
- [x] Fair comparison: Only architecture differs, not available features

## STRING Prior Details

The prior file (`tcga_string_prior.npz`) contains:
```python
{
    'A': adjacency_matrix,           # Binary PPI network
    'K': diffusion_kernel,           # Smoothed graph structure  
    'PE': positional_encodings,      # Graph Laplacian eigenvectors
    'protein_cols': protein_names,   # Column names
    'genes': gene_symbols           # Gene identifiers
}
```

Loaded by: `blinded_survival_classifier/data/graph_prior.py::load_graph_prior()`
