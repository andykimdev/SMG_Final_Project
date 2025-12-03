#!/usr/bin/env python3
"""
Analyze discordance between pathway activation (RPPA) and genomic mutations.

Research Question:
    Can we identify patients with hyper-active oncogenic signaling who lack
    the genomic mutations that typically drive that pathway?

These patients may:
    1. Have alternative activation mechanisms (e.g., upstream receptor activation)
    2. Benefit from targeted therapies despite lacking canonical mutations
    3. Have different prognosis than mutation-driven cases
"""

import pandas as pd
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from scipy import stats
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze pathway-mutation discordance"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="../processed_datasets/tcga_pancan_rppa_with_mutations.csv",
        help="Path to merged RPPA+mutations dataset"
    )
    parser.add_argument(
        "--pathways",
        type=str,
        default="../processed_datasets/mutation_matrix_pathways.json",
        help="Path to pathway definitions JSON"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/discordance_analysis",
        help="Output directory for results"
    )
    parser.add_argument(
        "--activation_threshold",
        type=float,
        default=0.5,
        help="Z-score threshold for pathway activation (default: 0.5 SD above mean)"
    )
    return parser.parse_args()


def define_rppa_pathway_signatures() -> Dict[str, List[str]]:
    """
    Define RPPA protein signatures for each pathway.

    Returns:
        Dictionary mapping pathway names to lists of RPPA protein column names
    """
    signatures = {
        'PI3K_AKT_MTOR': [
            'AKT1 AKT2 AKT3|Akt_pS473',       # Activated AKT
            'AKT1 AKT2 AKT3|Akt_pT308',       # Fully activated AKT
            'MTOR|mTOR_pS2448',                # Activated mTOR
            'RPS6KB1|p70S6K_pT389',            # Downstream of mTOR
            'RPS6|S6_pS235_S236',              # Ribosomal S6 (downstream)
            'RPS6|S6_pS240_S244',              # Alternative S6 phosphorylation
            'EIF4EBP1|4E-BP1_pS65',            # 4E-BP1 (mTOR target)
            'EIF4EBP1|4E-BP1_pT37T46',         # 4E-BP1 alternative sites
            'AKT1S1|PRAS40_pT246',             # mTOR substrate
            'GSK3A GSK3B|GSK3-alpha-beta_pS21_S9',  # AKT substrate (inactive when phosphorylated)
        ],
        'MAPK_ERK': [
            'MAPK1 MAPK3|MAPK_pT202_Y204',     # Phospho-ERK (direct readout)
            'MAP2K1|MEK1_pS217_S221',          # Phospho-MEK
            'RAF1|C-Raf_pS338',                # Phospho-C-Raf
            'RPS6KA1|p90RSK_pT359_S363',       # ERK substrate
            'RPS6KA1|p90RSK',                  # Total p90RSK
        ],
        'RTK_SIGNALING': [
            'EGFR|EGFR_pY1068',                # Activated EGFR
            'EGFR|EGFR_pY1173',                # Alternative EGFR phosphorylation
            'ERBB2|HER2_pY1248',               # Activated HER2
            'ERBB3|HER3_pY1298',               # Activated HER3
            'MET|c-Met_pY1235',                # Activated c-Met
            'SRC|Src_pY416',                   # Activated Src (RTK downstream)
            'SHC1|Shc_pY317',                  # Shc adaptor (RTK signaling)
        ],
        'CELL_CYCLE': [
            'CCND1|Cyclin_D1',                 # G1/S cyclin
            'CCNE1|Cyclin_E1',                 # S phase cyclin
            'CCNB1|Cyclin_B1',                 # M phase cyclin
            'CDK1|CDK1',                       # M-phase CDK
            'RB1|Rb_pS807_S811',               # Hyperphosphorylated Rb (inactive)
            'CDKN1A|p21',                      # CDK inhibitor
            'CDKN1B|p27',                      # CDK inhibitor
            'CDKN1B|p27_pT157',                # Inactivated p27
            'TP53|p53',                        # p53 tumor suppressor
        ],
        'WNT_BETA_CATENIN': [
            'CTNNB1|beta-Catenin',             # Beta-catenin (direct readout)
            'GSK3A GSK3B|GSK3-alpha-beta_pS21_S9',  # Inactive GSK3 (allows beta-catenin accumulation)
        ],
        'APOPTOSIS': [
            'BCL2|Bcl-2',                      # Anti-apoptotic
            'BCL2L1|Bcl-xL',                   # Anti-apoptotic
            'BAX|Bax',                         # Pro-apoptotic
            'BAK1|Bak',                        # Pro-apoptotic
            'CASP7|Caspase-7_cleavedD198',     # Activated caspase
            'PARP1|PARP1',                     # Apoptosis marker
        ],
    }

    return signatures


def compute_pathway_scores(df: pd.DataFrame, signatures: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Compute pathway activation scores from RPPA data.

    Args:
        df: DataFrame with RPPA protein columns
        signatures: Dictionary of pathway → protein list

    Returns:
        DataFrame with pathway score columns added
    """
    print("\nComputing pathway activation scores from RPPA...")

    for pathway_name, protein_list in signatures.items():
        # Find matching columns
        available_proteins = [p for p in protein_list if p in df.columns]
        missing_proteins = [p for p in protein_list if p not in df.columns]

        if missing_proteins:
            print(f"  {pathway_name}: {len(available_proteins)}/{len(protein_list)} proteins available")
            if len(missing_proteins) <= 3:
                print(f"    Missing: {', '.join(missing_proteins)}")

        if len(available_proteins) > 0:
            # Compute pathway score as mean of available protein z-scores
            pathway_score = df[available_proteins].mean(axis=1)
            df[f'PATHWAY_SCORE_{pathway_name}'] = pathway_score
        else:
            print(f"  WARNING: No proteins available for {pathway_name}")
            df[f'PATHWAY_SCORE_{pathway_name}'] = np.nan

    return df


def compute_pathway_mutations(df: pd.DataFrame, pathway_genes: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Compute pathway mutation status (any mutation in pathway genes).

    Args:
        df: DataFrame with MUT_GENE columns
        pathway_genes: Dictionary of pathway → gene list

    Returns:
        DataFrame with pathway mutation columns added
    """
    print("\nComputing pathway mutation status...")

    for pathway_name, gene_list in pathway_genes.items():
        # Find matching mutation columns
        mut_cols = [f'MUT_{gene}' for gene in gene_list if f'MUT_{gene}' in df.columns]

        if len(mut_cols) > 0:
            # Pathway has mutation if ANY gene is mutated
            df[f'HAS_MUT_{pathway_name}'] = (df[mut_cols].sum(axis=1) > 0).astype(int)
            n_mutated = df[f'HAS_MUT_{pathway_name}'].sum()
            pct = n_mutated / len(df) * 100
            print(f"  {pathway_name}: {n_mutated} patients ({pct:.1f}%) with mutations")
        else:
            print(f"  WARNING: No mutation data for {pathway_name}")
            df[f'HAS_MUT_{pathway_name}'] = 0

    return df


def identify_discordant_patients(
    df: pd.DataFrame,
    pathway_name: str,
    activation_threshold: float = 0.5
) -> Tuple[pd.DataFrame, Dict]:
    """
    Identify patients with pathway-mutation discordance.

    Defines 4 groups:
        1. HIGH SIGNALING + NO MUTATION ← Target group!
        2. LOW SIGNALING + HAS MUTATION ← Resistant?
        3. HIGH SIGNALING + HAS MUTATION ← Expected
        4. LOW SIGNALING + NO MUTATION ← Expected

    Args:
        df: DataFrame with pathway scores and mutations
        pathway_name: Name of pathway to analyze
        activation_threshold: Z-score threshold for "high" activation

    Returns:
        DataFrame with group assignment, statistics dictionary
    """
    score_col = f'PATHWAY_SCORE_{pathway_name}'
    mut_col = f'HAS_MUT_{pathway_name}'

    if score_col not in df.columns or mut_col not in df.columns:
        return df, {}

    # Define high/low activation based on z-score threshold
    pathway_zscore = (df[score_col] - df[score_col].mean()) / df[score_col].std()
    high_activation = (pathway_zscore > activation_threshold).astype(int)
    has_mutation = df[mut_col]

    # Assign groups
    group = np.zeros(len(df), dtype=int)
    group[(high_activation == 1) & (has_mutation == 0)] = 1  # HIGH + NO MUT
    group[(high_activation == 0) & (has_mutation == 1)] = 2  # LOW + HAS MUT
    group[(high_activation == 1) & (has_mutation == 1)] = 3  # HIGH + HAS MUT
    group[(high_activation == 0) & (has_mutation == 0)] = 4  # LOW + NO MUT

    df[f'GROUP_{pathway_name}'] = group

    # Compute statistics
    stats_dict = {
        'pathway': pathway_name,
        'n_high_activation_no_mut': int((group == 1).sum()),
        'n_low_activation_has_mut': int((group == 2).sum()),
        'n_high_activation_has_mut': int((group == 3).sum()),
        'n_low_activation_no_mut': int((group == 4).sum()),
        'pct_high_activation_no_mut': float((group == 1).sum() / len(df) * 100),
        'pct_low_activation_has_mut': float((group == 2).sum() / len(df) * 100),
        'mean_score_mutated': float(df[has_mutation == 1][score_col].mean()),
        'mean_score_wildtype': float(df[has_mutation == 0][score_col].mean()),
        'score_diff_pvalue': float(stats.ttest_ind(
            df[has_mutation == 1][score_col].dropna(),
            df[has_mutation == 0][score_col].dropna()
        )[1]) if (has_mutation.sum() > 0 and (1-has_mutation).sum() > 0) else np.nan,
    }

    return df, stats_dict


def analyze_survival_by_group(
    df: pd.DataFrame,
    pathway_name: str,
    time_col: str = 'DSS_MONTHS',
    event_col: str = 'DSS_STATUS'
) -> Dict:
    """
    Compare survival outcomes across concordance groups.

    Args:
        df: DataFrame with group assignments
        pathway_name: Name of pathway
        time_col: Survival time column
        event_col: Survival event column (0=censored, 1=event)

    Returns:
        Dictionary with survival statistics
    """
    group_col = f'GROUP_{pathway_name}'

    if group_col not in df.columns:
        return {}

    # Filter for valid survival data
    valid = df[time_col].notna() & df[event_col].notna() & (df[time_col] > 0)
    df_surv = df[valid].copy()

    # Convert DSS_STATUS to numeric if needed
    if df_surv[event_col].dtype == 'object':
        df_surv[event_col] = df_surv[event_col].apply(
            lambda x: 1 if ('1' in str(x) or 'DECEASED' in str(x).upper()) else 0
        )

    results = {
        'pathway': pathway_name,
        'n_patients_with_survival': len(df_surv),
        'groups': {}
    }

    # Analyze each group
    for group_id in [1, 2, 3, 4]:
        group_mask = df_surv[group_col] == group_id
        n = group_mask.sum()

        if n > 0:
            group_data = df_surv[group_mask]
            n_events = group_data[event_col].sum()
            median_time = group_data[time_col].median()

            group_name = {
                1: 'HIGH_SIGNAL_NO_MUT',
                2: 'LOW_SIGNAL_HAS_MUT',
                3: 'HIGH_SIGNAL_HAS_MUT',
                4: 'LOW_SIGNAL_NO_MUT'
            }[group_id]

            results['groups'][group_name] = {
                'n': int(n),
                'n_events': int(n_events),
                'event_rate': float(n_events / n),
                'median_survival_months': float(median_time)
            }

    # Log-rank test: Group 1 vs Group 3 (both high signal, with/without mutation)
    if ('HIGH_SIGNAL_NO_MUT' in results['groups'] and
        'HIGH_SIGNAL_HAS_MUT' in results['groups']):
        group1 = df_surv[df_surv[group_col] == 1]
        group3 = df_surv[df_surv[group_col] == 3]

        if len(group1) > 0 and len(group3) > 0:
            lr_result = logrank_test(
                group1[time_col], group3[time_col],
                group1[event_col], group3[event_col]
            )
            results['logrank_high_signal_mut_vs_nomut'] = {
                'p_value': float(lr_result.p_value),
                'test_statistic': float(lr_result.test_statistic)
            }

    return results


def plot_pathway_discordance(
    df: pd.DataFrame,
    pathway_name: str,
    output_dir: str
):
    """
    Create visualization of pathway-mutation discordance.

    Args:
        df: DataFrame with pathway scores and mutations
        pathway_name: Name of pathway
        output_dir: Output directory for plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    score_col = f'PATHWAY_SCORE_{pathway_name}'
    mut_col = f'HAS_MUT_{pathway_name}'

    if score_col not in df.columns or mut_col not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Pathway score distribution by mutation status
    mutated = df[df[mut_col] == 1][score_col].dropna()
    wildtype = df[df[mut_col] == 0][score_col].dropna()

    axes[0].hist(wildtype, bins=50, alpha=0.6, label='Wild-type', color='blue')
    axes[0].hist(mutated, bins=50, alpha=0.6, label='Mutated', color='red')
    axes[0].axvline(wildtype.mean(), color='blue', linestyle='--', linewidth=2)
    axes[0].axvline(mutated.mean(), color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Pathway Activation Score (RPPA)')
    axes[0].set_ylabel('Number of Patients')
    axes[0].set_title(f'{pathway_name}: Signaling vs Mutation Status')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot 2: Scatter plot showing discordance
    group_col = f'GROUP_{pathway_name}'
    if group_col in df.columns:
        colors = {1: 'orange', 2: 'purple', 3: 'red', 4: 'blue'}
        labels = {
            1: 'High Signal, No Mut (TARGET)',
            2: 'Low Signal, Has Mut',
            3: 'High Signal, Has Mut',
            4: 'Low Signal, No Mut'
        }

        for group_id in [4, 3, 2, 1]:  # Plot in reverse order so target group is on top
            mask = df[group_col] == group_id
            if mask.sum() > 0:
                axes[1].scatter(
                    df[mask][mut_col] + np.random.normal(0, 0.02, mask.sum()),
                    df[mask][score_col],
                    alpha=0.5,
                    s=20,
                    c=colors[group_id],
                    label=f"{labels[group_id]} (n={mask.sum()})"
                )

        axes[1].axhline(0, color='gray', linestyle='--', linewidth=1)
        axes[1].set_xlabel('Mutation Status')
        axes[1].set_ylabel('Pathway Activation Score (RPPA)')
        axes[1].set_xticks([0, 1])
        axes[1].set_xticklabels(['Wild-type', 'Mutated'])
        axes[1].set_title(f'{pathway_name}: Discordance Groups')
        axes[1].legend(loc='best', fontsize=8)
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{pathway_name}_discordance.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved plot: {output_dir}/{pathway_name}_discordance.png")


def main():
    args = parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("Pathway-Mutation Discordance Analysis")
    print("="*80)

    # Load data
    print(f"\nLoading data from {args.data}...")
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} patients")

    # Load pathway definitions
    print(f"Loading pathway definitions from {args.pathways}...")
    with open(args.pathways, 'r') as f:
        pathway_genes = json.load(f)

    # Define RPPA signatures
    rppa_signatures = define_rppa_pathway_signatures()

    # Compute pathway scores from RPPA
    df = compute_pathway_scores(df, rppa_signatures)

    # Compute pathway mutation status
    df = compute_pathway_mutations(df, pathway_genes)

    # Analyze each pathway
    all_stats = []
    all_survival = []

    for pathway_name in pathway_genes.keys():
        print(f"\n{'='*80}")
        print(f"Analyzing {pathway_name}")
        print(f"{'='*80}")

        # Identify discordant patients
        df, stats = identify_discordant_patients(
            df, pathway_name, args.activation_threshold
        )

        if stats:
            all_stats.append(stats)
            print(f"\nDiscordance groups:")
            print(f"  HIGH signaling + NO mutation: {stats['n_high_activation_no_mut']} ({stats['pct_high_activation_no_mut']:.1f}%)")
            print(f"  LOW signaling + HAS mutation: {stats['n_low_activation_has_mut']} ({stats['pct_low_activation_has_mut']:.1f}%)")
            print(f"  HIGH signaling + HAS mutation: {stats['n_high_activation_has_mut']}")
            print(f"  LOW signaling + NO mutation: {stats['n_low_activation_no_mut']}")
            print(f"\nMean pathway scores:")
            print(f"  Mutated patients: {stats['mean_score_mutated']:.3f}")
            print(f"  Wild-type patients: {stats['mean_score_wildtype']:.3f}")
            print(f"  P-value (t-test): {stats['score_diff_pvalue']:.4f}")

        # Analyze survival
        surv_results = analyze_survival_by_group(df, pathway_name)
        if surv_results:
            all_survival.append(surv_results)
            print(f"\nSurvival analysis:")
            for group_name, group_stats in surv_results['groups'].items():
                print(f"  {group_name}: n={group_stats['n']}, "
                      f"events={group_stats['n_events']} ({group_stats['event_rate']*100:.1f}%), "
                      f"median={group_stats['median_survival_months']:.1f} months")

            if 'logrank_high_signal_mut_vs_nomut' in surv_results:
                lr = surv_results['logrank_high_signal_mut_vs_nomut']
                print(f"  Log-rank test (HIGH_SIGNAL: Mut vs No Mut): p={lr['p_value']:.4f}")

        # Create visualization
        plot_pathway_discordance(df, pathway_name, args.output_dir)

    # Save results
    stats_df = pd.DataFrame(all_stats)
    stats_output = f"{args.output_dir}/discordance_statistics.csv"
    stats_df.to_csv(stats_output, index=False)
    print(f"\n\nSaved discordance statistics to {stats_output}")

    survival_output = f"{args.output_dir}/survival_by_discordance.json"
    with open(survival_output, 'w') as f:
        json.dump(all_survival, f, indent=2)
    print(f"Saved survival results to {survival_output}")

    # Save annotated dataset
    annotated_output = f"{args.output_dir}/patients_with_discordance_groups.csv"
    df.to_csv(annotated_output, index=False)
    print(f"Saved annotated patient data to {annotated_output}")

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
