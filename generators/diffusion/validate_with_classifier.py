"""
Validation Script: Classifier Consistency Check.

This script evaluates the quality of the generative model by:
1. Creating 1000 realistic synthetic patient profiles using ACTUAL training data distributions
2. Generating protein expression profiles for them using the Diffusion Model
3. Feeding the generated proteins into the pre-trained Classifier
4. Checking if the classifier predicts the correct cancer type

Key Features:
- Samples from per-cancer-type distributions for realistic contexts
- Includes survival outcomes in context generation
- Enforces biological constraints (e.g., BRCA → Female)
- Uses proper rounding and units from training data

UPDATED: Proper per-cancer sampling, survival outcomes included.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path
from collections import defaultdict

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Cancer Classifier'))
sys.path.append(os.path.dirname(__file__))

import config
from diffusion_model import create_diffusion_model
from diffusion_utils import GaussianDiffusion, EMA
from graph_transformer_classifier import GraphTransformerClassifier
from graph_prior import load_graph_prior


def compute_per_cancer_statistics(csv_path: str, context_info: dict) -> dict:
    """
    Compute per-cancer-type statistics from the training data.
    
    This extracts the actual distributions of:
    - Age (mean, std per cancer type)
    - Sex proportions per cancer type
    - Stage distribution per cancer type
    - Molecular scores (mean, std per cancer type)
    - Survival outcomes distribution per cancer type
    
    Returns:
        stats: Dictionary with per-cancer statistics
    """
    print("Computing per-cancer-type statistics from training data...")
    
    df = pd.read_csv(csv_path)
    
    # Filter to valid cancer types
    cancer_types = list(context_info['cancer_type_encoder'].classes_)
    df = df[df['CANCER_TYPE_ACRONYM'].isin(cancer_types)].copy()
    
    stats = {}
    
    for cancer in cancer_types:
        cancer_df = df[df['CANCER_TYPE_ACRONYM'] == cancer]
        
        if len(cancer_df) == 0:
            continue
        
        cancer_stats = {}
        
        # Age statistics
        if 'AGE' in cancer_df.columns:
            ages = cancer_df['AGE'].dropna()
            if len(ages) > 0:
                cancer_stats['age_mean'] = ages.mean()
                cancer_stats['age_std'] = max(ages.std(), 5.0)  # Min std of 5 years
            else:
                cancer_stats['age_mean'] = 60.0
                cancer_stats['age_std'] = 12.0
        else:
            cancer_stats['age_mean'] = 60.0
            cancer_stats['age_std'] = 12.0
        
        # Sex proportions
        if 'SEX' in cancer_df.columns:
            sex_counts = cancer_df['SEX'].value_counts(normalize=True)
            cancer_stats['male_prob'] = sex_counts.get('Male', 0.5)
            cancer_stats['female_prob'] = sex_counts.get('Female', 0.5)
        else:
            cancer_stats['male_prob'] = 0.5
            cancer_stats['female_prob'] = 0.5
        
        # Stage distribution
        if 'AJCC_PATHOLOGIC_TUMOR_STAGE' in cancer_df.columns:
            stage_counts = cancer_df['AJCC_PATHOLOGIC_TUMOR_STAGE'].value_counts(normalize=True)
            cancer_stats['stage_probs'] = stage_counts.to_dict()
        else:
            cancer_stats['stage_probs'] = {'Unknown': 1.0}
        
        # Molecular scores (raw, not standardized)
        mol_features = ['ANEUPLOIDY_SCORE', 'TMB_NONSYNONYMOUS', 'MSI_SCORE_MANTIS', 'TBL_SCORE']
        mol_stats = {}
        for feat in mol_features:
            if feat in cancer_df.columns:
                values = cancer_df[feat].dropna()
                if len(values) > 0:
                    mol_stats[feat] = {'mean': values.mean(), 'std': max(values.std(), 0.1)}
                else:
                    mol_stats[feat] = {'mean': 0.0, 'std': 1.0}
            else:
                mol_stats[feat] = {'mean': 0.0, 'std': 1.0}
        cancer_stats['molecular'] = mol_stats
        
        # Survival outcomes
        survival_outcomes = [
            ('OS_STATUS', 'OS_MONTHS'),
            ('PFS_STATUS', 'PFS_MONTHS'),
            ('DSS_STATUS', 'DSS_MONTHS'),
            ('DFS_STATUS', 'DFS_MONTHS'),
        ]
        
        survival_stats = {}
        for status_col, months_col in survival_outcomes:
            prefix = status_col.split('_')[0].lower()
            
            # Status distribution
            if status_col in cancer_df.columns:
                # Parse status
                def parse_status(s):
                    if pd.isna(s) or str(s).strip() == '':
                        return 2
                    s = str(s).upper()
                    if any(x in s for x in ['1:', 'DECEASED', 'DEAD', 'PROGRESSION', 'RECURRED']):
                        return 1
                    if any(x in s for x in ['0:', 'LIVING', 'ALIVE', 'CENSORED', 'DISEASEFREE']):
                        return 0
                    return 2
                
                statuses = cancer_df[status_col].apply(parse_status)
                status_counts = statuses.value_counts(normalize=True)
                survival_stats[f'{prefix}_status_probs'] = {
                    0: status_counts.get(0, 0.33),
                    1: status_counts.get(1, 0.33),
                    2: status_counts.get(2, 0.34),
                }
            else:
                survival_stats[f'{prefix}_status_probs'] = {0: 0.33, 1: 0.33, 2: 0.34}
            
            # Months statistics
            if months_col in cancer_df.columns:
                months = pd.to_numeric(cancer_df[months_col], errors='coerce').dropna()
                if len(months) > 0:
                    survival_stats[f'{prefix}_months_mean'] = months.mean()
                    survival_stats[f'{prefix}_months_std'] = max(months.std(), 6.0)
                    survival_stats[f'{prefix}_months_max'] = months.max()
                else:
                    survival_stats[f'{prefix}_months_mean'] = 36.0
                    survival_stats[f'{prefix}_months_std'] = 24.0
                    survival_stats[f'{prefix}_months_max'] = 120.0
            else:
                survival_stats[f'{prefix}_months_mean'] = 36.0
                survival_stats[f'{prefix}_months_std'] = 24.0
                survival_stats[f'{prefix}_months_max'] = 120.0
        
        cancer_stats['survival'] = survival_stats
        
        stats[cancer] = cancer_stats
    
    print(f"  Computed statistics for {len(stats)} cancer types")
    return stats


def create_realistic_contexts_from_distributions(
    num_samples: int,
    context_info: dict,
    per_cancer_stats: dict,
    device: torch.device
) -> tuple:
    """
    Create realistic patient contexts by sampling from per-cancer-type distributions.
    
    This ensures:
    - Age is sampled from the cancer-specific distribution
    - Sex respects biological constraints (BRCA→Female, PRAD→Male)
    - Stage is sampled from cancer-specific stage distribution
    - Molecular scores are sampled from cancer-specific distributions
    - Survival outcomes are sampled from cancer-specific distributions
    
    Returns:
        context_batch: Dictionary of tensors for model input
        true_labels: List of cancer type strings
    """
    cancer_types = list(context_info['cancer_type_encoder'].classes_)
    stages = list(context_info['stage_encoder'].classes_)
    sexes = list(context_info['sex_encoder'].classes_)
    age_min = context_info['age_min']
    age_max = context_info['age_max']
    
    print(f"\n=== Training Data Categories ===")
    print(f"Cancer types ({len(cancer_types)}): {cancer_types}")
    print(f"Stages ({len(stages)}): {stages}")
    print(f"Sexes: {sexes}")
    print(f"Age range: {age_min:.0f} - {age_max:.0f}")
    print()
    
    # Female-only and male-only cancers
    FEMALE_CANCERS = {'BRCA', 'UCEC', 'CESC', 'OV', 'UCS'}
    MALE_CANCERS = {'PRAD', 'TGCT'}
    
    # Storage
    cancer_type_indices = []
    stage_indices = []
    ages = []
    sex_indices = []
    molecular_scores = []
    os_statuses = []
    os_months = []
    pfs_statuses = []
    pfs_months = []
    dss_statuses = []
    dss_months = []
    dfs_statuses = []
    dfs_months = []
    true_labels = []
    
    print(f"Generating {num_samples} realistic synthetic contexts...")
    
    for i in range(num_samples):
        # 1. Cancer type: balanced across all types
        cancer_type = cancer_types[i % len(cancer_types)]
        cancer_idx = context_info['cancer_type_encoder'].transform([cancer_type])[0]
        
        # Get per-cancer stats (or use defaults)
        stats = per_cancer_stats.get(cancer_type, {})
        
        # 2. Age: sample from cancer-specific distribution
        age_mean = stats.get('age_mean', 60.0)
        age_std = stats.get('age_std', 12.0)
        age = np.clip(np.random.normal(age_mean, age_std), age_min, age_max)
        age = round(age, 1)  # Round to 1 decimal like training data
        age_normalized = (age - age_min) / (age_max - age_min + 1e-8)
        
        # 3. Sex: respect biological constraints
        if cancer_type in FEMALE_CANCERS:
            sex = 'Female'
        elif cancer_type in MALE_CANCERS:
            sex = 'Male'
        else:
            male_prob = stats.get('male_prob', 0.5)
            sex = 'Male' if np.random.random() < male_prob else 'Female'
        
        try:
            sex_idx = context_info['sex_encoder'].transform([sex])[0]
        except ValueError:
            sex_idx = 0
        
        # 4. Stage: sample from cancer-specific distribution
        stage_probs = stats.get('stage_probs', {'Unknown': 1.0})
        available_stages = [s for s in stage_probs.keys() if s in stages]
        
        if available_stages:
            probs = np.array([stage_probs.get(s, 0) for s in available_stages])
            probs = probs / probs.sum()
            stage = np.random.choice(available_stages, p=probs)
        else:
            stage = 'Unknown'
        
        try:
            stage_idx = context_info['stage_encoder'].transform([stage])[0]
        except ValueError:
            stage_idx = context_info['stage_encoder'].transform(['Unknown'])[0]
        
        # 5. Molecular scores: sample from cancer-specific distributions
        mol_stats = stats.get('molecular', {})
        mol_values = []
        for feat in ['ANEUPLOIDY_SCORE', 'TMB_NONSYNONYMOUS', 'MSI_SCORE_MANTIS', 'TBL_SCORE']:
            feat_stats = mol_stats.get(feat, {'mean': 0, 'std': 1})
            val = np.random.normal(feat_stats['mean'], feat_stats['std'])
            mol_values.append(val)
        
        # Standardize molecular scores using the training scaler
        mol_values = np.array(mol_values).reshape(1, -1)
        mol_values_scaled = context_info['molecular_scaler'].transform(mol_values)[0]
        
        # 6. Survival outcomes: sample from cancer-specific distributions
        surv_stats = stats.get('survival', {})
        
        for prefix in ['os', 'pfs', 'dss', 'dfs']:
            # Status - get probabilities and normalize to sum to 1
            status_probs = surv_stats.get(f'{prefix}_status_probs', {0: 0.33, 1: 0.33, 2: 0.34})
            p0 = status_probs.get(0, 0.33)
            p1 = status_probs.get(1, 0.33)
            p2 = status_probs.get(2, 0.34)
            total = p0 + p1 + p2
            if total == 0:
                total = 1.0
                p0, p1, p2 = 0.33, 0.33, 0.34
            status = np.random.choice([0, 1, 2], p=[p0/total, p1/total, p2/total])
            
            # Months
            months_mean = surv_stats.get(f'{prefix}_months_mean', 36.0)
            months_std = surv_stats.get(f'{prefix}_months_std', 24.0)
            months_max = surv_stats.get(f'{prefix}_months_max', 120.0)
            
            months_val = np.clip(np.random.normal(months_mean, months_std), 0, months_max * 1.5)
            months_normalized = months_val / (months_max + 1e-8)
            months_normalized = np.clip(months_normalized, 0, 1)
            
            if prefix == 'os':
                os_statuses.append(status)
                os_months.append(months_normalized)
            elif prefix == 'pfs':
                pfs_statuses.append(status)
                pfs_months.append(months_normalized)
            elif prefix == 'dss':
                dss_statuses.append(status)
                dss_months.append(months_normalized)
            else:
                dfs_statuses.append(status)
                dfs_months.append(months_normalized)
        
        cancer_type_indices.append(cancer_idx)
        stage_indices.append(stage_idx)
        ages.append(age_normalized)
        sex_indices.append(sex_idx)
        molecular_scores.append(mol_values_scaled)
        true_labels.append(cancer_type)
    
    # Convert to tensors
    context_batch = {
        'cancer_type': torch.tensor(cancer_type_indices, dtype=torch.long, device=device),
        'stage': torch.tensor(stage_indices, dtype=torch.long, device=device),
        'age': torch.tensor(ages, dtype=torch.float32, device=device).unsqueeze(-1),
        'sex': torch.tensor(sex_indices, dtype=torch.long, device=device),
        'molecular': torch.tensor(np.array(molecular_scores), dtype=torch.float32, device=device),
        'os_status': torch.tensor(os_statuses, dtype=torch.long, device=device),
        'os_months': torch.tensor(os_months, dtype=torch.float32, device=device).unsqueeze(-1),
        'pfs_status': torch.tensor(pfs_statuses, dtype=torch.long, device=device),
        'pfs_months': torch.tensor(pfs_months, dtype=torch.float32, device=device).unsqueeze(-1),
        'dss_status': torch.tensor(dss_statuses, dtype=torch.long, device=device),
        'dss_months': torch.tensor(dss_months, dtype=torch.float32, device=device).unsqueeze(-1),
        'dfs_status': torch.tensor(dfs_statuses, dtype=torch.long, device=device),
        'dfs_months': torch.tensor(dfs_months, dtype=torch.float32, device=device).unsqueeze(-1),
    }
    
    print(f"  Generated {num_samples} contexts")
    print(f"  Cancer type distribution: ~{num_samples // len(cancer_types)} per type")
    
    return context_batch, true_labels


@torch.no_grad()
def main():
    # Setup
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path("validation_results")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Load Graph Prior
    print("\n" + "="*60)
    print("STEP 1: Loading Graph Prior")
    print("="*60)
    prior_path = "../priors/tcga_string_prior.npz"
    graph_prior = load_graph_prior(prior_path)
    print(f"  Loaded graph prior with {graph_prior['K'].shape[0]} proteins")
    
    # 2. Load Diffusion Model
    print("\n" + "="*60)
    print("STEP 2: Loading Diffusion Model")
    print("="*60)
    diff_checkpoint_path = "outputs/checkpoints/best_model.pt"
    if not os.path.exists(diff_checkpoint_path):
        print("❌ Error: Diffusion checkpoint not found. Train it first!")
        return

    diff_checkpoint = torch.load(diff_checkpoint_path, map_location=device, weights_only=False)
    context_info = diff_checkpoint['context_info']
    
    diffusion_model = create_diffusion_model(
        graph_prior=graph_prior,
        num_cancer_types=context_info['num_cancer_types'],
        num_stages=context_info['num_stages'],
        num_sexes=context_info['num_sexes'],
        num_survival_status=context_info.get('num_survival_status', 3),
        load_classifier=False,
        device=device
    )
    diffusion_model.load_state_dict(diff_checkpoint['model_state_dict'])
    diffusion_model.eval()
    print(f"  Loaded diffusion model from epoch {diff_checkpoint.get('epoch', 'unknown')}")
    
    # Use EMA if available
    if 'ema_state_dict' in diff_checkpoint:
        print("  Using EMA weights for generation (higher quality)")
        ema = EMA(diffusion_model)
        ema.load_state_dict(diff_checkpoint['ema_state_dict'])
        ema.apply_shadow(diffusion_model)
        
    diffusion = GaussianDiffusion(
        timesteps=config.DIFFUSION['timesteps'],
        schedule=config.DIFFUSION['schedule']
    )
    
    # 3. Load Classifier
    print("\n" + "="*60)
    print("STEP 3: Loading Classifier")
    print("="*60)
    clf_checkpoint_path = "../Cancer Classifier/outputs/checkpoints/best_model.pt"
    clf_checkpoint = torch.load(clf_checkpoint_path, map_location=device, weights_only=False)
    label_info = clf_checkpoint['label_info']
    
    print(f"  Classifier has {label_info['n_classes']} classes")
    print(f"  Classes: {label_info['class_names']}")
    
    # Reconstruct classifier with its ORIGINAL architecture (128 dim, 4 layers)
    # NOT the diffusion model's architecture (256 dim, 6 layers)
    K_tensor = torch.from_numpy(graph_prior['K']).to(device)
    PE_tensor = torch.from_numpy(graph_prior['PE']).to(device)
    
    # Get classifier architecture from checkpoint if available, else use defaults
    clf_config = clf_checkpoint.get('config', {})
    clf_model_config = clf_config.get('MODEL', {})
    
    classifier = GraphTransformerClassifier(
        n_proteins=198,
        n_classes=label_info['n_classes'],
        diffusion_kernel=K_tensor,
        positional_encodings=PE_tensor,
        # Use classifier's original architecture from checkpoint, not Generator's config
        embedding_dim=clf_model_config.get('embedding_dim', 128),
        n_layers=clf_model_config.get('n_layers', 4),
        n_heads=clf_model_config.get('n_heads', 8),
        ffn_dim=clf_model_config.get('ffn_dim', 512),  # Critical: use classifier's ffn_dim
        dropout=clf_model_config.get('dropout', 0.1),
    ).to(device)
    classifier.load_state_dict(clf_checkpoint['model_state_dict'])
    classifier.eval()
    
    # 4. Compute per-cancer statistics from training data
    print("\n" + "="*60)
    print("STEP 4: Computing Per-Cancer Statistics")
    print("="*60)
    csv_path = config.PATHS['csv_path']
    per_cancer_stats = compute_per_cancer_statistics(csv_path, context_info)
    
    # 5. Generate Synthetic Contexts
    print("\n" + "="*60)
    print("STEP 5: Creating Realistic Biological Contexts")
    print("="*60)
    
    num_samples = 1000
    context_batch, true_labels = create_realistic_contexts_from_distributions(
        num_samples=num_samples,
        context_info=context_info,
        per_cancer_stats=per_cancer_stats,
        device=device
    )
    
    # 6. Generate Protein Profiles
    print("\n" + "="*60)
    print("STEP 6: Generating Protein Profiles via Diffusion")
    print("="*60)
    
    generated_proteins = []
    batch_size = 32
    
    for i in tqdm(range(0, num_samples, batch_size), desc="Generating batches"):
        end_idx = min(i + batch_size, num_samples)
        
        # Slice context batch
        batch_ctx = {
            key: val[i:end_idx] for key, val in context_batch.items()
        }
        
        # Sample from diffusion model
        samples = diffusion.p_sample_loop(
            diffusion_model,
            shape=(end_idx - i, 198),
            context_dict=batch_ctx,
            device=device,
            progress=False
        )
        generated_proteins.append(samples)
        
    generated_proteins = torch.cat(generated_proteins, dim=0)
    print(f"  Generated {generated_proteins.shape[0]} protein profiles")
    print(f"  Protein profile shape: {generated_proteins.shape[1]} dimensions")
    print(f"  Sample stats - Mean: {generated_proteins.mean():.3f}, Std: {generated_proteins.std():.3f}")
    
    # 7. Classify Generated Data
    print("\n" + "="*60)
    print("STEP 7: Classifying Generated Profiles")
    print("="*60)
    
    logits = classifier(generated_proteins)
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    
    # Convert true labels to indices using classifier's mapping
    label_to_idx = label_info['label_to_idx']
    true_indices = torch.tensor([label_to_idx[l] for l in true_labels], device=device)
    
    # 8. Compute Metrics
    print("\n" + "="*60)
    print("STEP 8: Computing Metrics")
    print("="*60)
    
    acc = accuracy_score(true_indices.cpu().numpy(), preds.cpu().numpy())
    
    print(f"\n{'='*60}")
    print(f"✅ CLASSIFIER CONSISTENCY ACCURACY: {acc:.4f} ({acc*100:.1f}%)")
    print(f"{'='*60}")
    print(f"\nInterpretation:")
    print(f"  - Random chance baseline: {1/label_info['n_classes']*100:.1f}%")
    print(f"  - Actual accuracy: {acc*100:.1f}%")
    
    if acc > 0.5:
        print(f"  - Result: GOOD - Model captures cancer-specific protein patterns")
    elif acc > 0.2:
        print(f"  - Result: MODERATE - Model partially captures patterns")
    else:
        print(f"  - Result: POOR - Model needs more training")
    
    # 9. Save Results
    print("\n" + "="*60)
    print("STEP 9: Saving Results")
    print("="*60)
    
    # Save metrics text
    report = classification_report(
        true_indices.cpu().numpy(), 
        preds.cpu().numpy(),
        target_names=label_info['class_names'],
        zero_division=0
    )
    
    with open(output_dir / "consistency_metrics.txt", "w") as f:
        f.write("="*60 + "\n")
        f.write("CLASSIFIER CONSISTENCY VALIDATION RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Classifier Consistency Accuracy: {acc:.4f}\n")
        f.write(f"Number of samples: {num_samples}\n")
        f.write(f"Random baseline: {1/label_info['n_classes']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print(f"  Saved metrics to {output_dir}/consistency_metrics.txt")
    
    # Save generated samples for inspection
    np.savez(
        output_dir / "generated_samples.npz",
        proteins=generated_proteins.cpu().numpy(),
        true_labels=np.array(true_labels),
        pred_labels=np.array([label_info['class_names'][i] for i in preds.cpu().numpy()]),
        probabilities=probs.cpu().numpy()
    )
    print(f"  Saved generated samples to {output_dir}/generated_samples.npz")
    
    # Confusion Matrix Plot
    cm = confusion_matrix(true_indices.cpu().numpy(), preds.cpu().numpy())
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d',
        cmap='Blues',
        xticklabels=label_info['class_names'], 
        yticklabels=label_info['class_names'],
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted (by Classifier)', fontsize=12)
    plt.ylabel('True (Conditioned Context)', fontsize=12)
    plt.title(f'Generator-Classifier Consistency\nAccuracy: {acc:.3f} (Baseline: {1/label_info["n_classes"]:.3f})', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "consistency_confusion_matrix.png", dpi=300)
    print(f"  Saved confusion matrix to {output_dir}/consistency_confusion_matrix.png")
    
    # Per-class accuracy plot
    class_correct = {}
    class_total = {}
    for i, (true_idx, pred_idx) in enumerate(zip(true_indices.cpu().numpy(), preds.cpu().numpy())):
        label = label_info['class_names'][true_idx]
        class_total[label] = class_total.get(label, 0) + 1
        if true_idx == pred_idx:
            class_correct[label] = class_correct.get(label, 0) + 1
    
    class_acc = {k: class_correct.get(k, 0) / class_total[k] for k in class_total}
    
    plt.figure(figsize=(14, 6))
    classes = list(class_acc.keys())
    accs = [class_acc[c] for c in classes]
    colors = ['green' if a > 0.5 else 'orange' if a > 0.2 else 'red' for a in accs]
    
    plt.bar(range(len(classes)), accs, color=colors)
    plt.axhline(y=1/label_info['n_classes'], color='gray', linestyle='--', label='Random baseline')
    plt.axhline(y=acc, color='blue', linestyle='-', alpha=0.7, label=f'Overall acc: {acc:.3f}')
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    plt.ylabel('Accuracy')
    plt.xlabel('Cancer Type')
    plt.title('Per-Class Classifier Consistency Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "per_class_accuracy.png", dpi=300)
    print(f"  Saved per-class accuracy to {output_dir}/per_class_accuracy.png")
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
    print(f"\nAll results saved to: {output_dir.absolute()}/")


if __name__ == "__main__":
    main()
