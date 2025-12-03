"""
Linear baseline models for survival prediction.
Tests if simple linear methods (PCA + Cox, Elastic Net Cox) can match neural networks.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import torch

import config
from graph_prior import load_graph_prior
from dataset_survival_classifier import load_and_preprocess_survival_data


def prepare_survival_data(data_splits, survival_info):
    """Prepare train/val/test feature arrays and survival DataFrames."""
    print(f"\nPreparing survival data...")
    
    # Debug: check what's in the tuple BEFORE trying to unpack
    train_tuple = data_splits['train']
    print(f"DEBUG: train tuple length: {len(train_tuple)}")
    print(f"DEBUG: train tuple element shapes: {[x.shape if hasattr(x, 'shape') else type(x) for x in train_tuple]}")
    
    # Based on the shapes, figure out which are features (2D) and which are labels (1D)
    # Likely: (protein_features, clinical_features, genomic_features, event, time)
    # OR: (all_features, protein_only, event, time, something_else)
    
    def extract_data(split_tuple):
        """Extract features and labels from split tuple."""
        
        if len(split_tuple) == 5:
            # Print shapes to understand structure
            shapes = [x.shape for x in split_tuple]
            print(f"  5-element tuple with shapes: {shapes}")
            
            # Find the 2D array with most features (that's X)
            # Find 1D arrays for event and time
            arrays_2d = [(i, x) for i, x in enumerate(split_tuple) if len(x.shape) == 2]
            arrays_1d = [(i, x) for i, x in enumerate(split_tuple) if len(x.shape) == 1]
            
            if len(arrays_2d) > 0:
                # Use the first/largest 2D array as features
                X = arrays_2d[0][1]
            else:
                raise ValueError("No 2D array found for features")
            
            if len(arrays_1d) >= 2:
                # Assume last two 1D arrays are event and time
                event = arrays_1d[-2][1]
                time = arrays_1d[-1][1]
            else:
                raise ValueError(f"Need at least 2 1D arrays for event/time, found {len(arrays_1d)}")
                
        elif len(split_tuple) == 3:
            # (X, event, time)
            X, event, time = split_tuple
                
        elif len(split_tuple) == 2:
            X, y = split_tuple
            # y might be a dict or tuple with event and time
            if isinstance(y, dict):
                event = y['event']
                time = y['time']
            elif isinstance(y, (tuple, list)) and len(y) == 2:
                event, time = y
            else:
                raise ValueError(f"Unexpected y format: {type(y)}")
        else:
            raise ValueError(f"Unexpected split_tuple length: {len(split_tuple)}")
        
        # Create DataFrame
        y_df = pd.DataFrame({
            'event': event,
            'time': time
        })
        
        return X, y_df
    
    # Extract train/val/test
    train_X, train_y = extract_data(data_splits['train'])
    val_X, val_y = extract_data(data_splits['val'])
    test_X, test_y = extract_data(data_splits['test'])
    
    print(f"Train: {train_X.shape[0]} samples, {train_X.shape[1]} features")
    print(f"Val:   {val_X.shape[0]} samples")
    print(f"Test:  {test_X.shape[0]} samples")
    
    return (train_X, train_y), (val_X, val_y), (test_X, test_y)


def train_cox_pca(train_data, val_data, test_data, n_components=50):
    """
    Train Cox PH with PCA dimensionality reduction.

    Tests: Can linear dimensionality reduction capture survival signals?
    """
    train_X, train_y = train_data
    val_X, val_y = val_data
    test_X, test_y = test_data

    print(f"\n{'='*80}")
    print(f"LINEAR BASELINE 1: Cox PH + PCA ({n_components} components)")
    print(f"{'='*80}")

    # Fit scaler and PCA
    scaler = StandardScaler()
    pca = PCA(n_components=n_components)
    
    print(f"Fitting PCA with {n_components} components...")
    X_train_scaled = scaler.fit_transform(train_X)
    X_train_pca = pca.fit_transform(X_train_scaled)
    
    # Create DataFrame for Cox model
    df_train = pd.DataFrame(X_train_pca, columns=[f'PC{i}' for i in range(n_components)])
    df_train['time'] = train_y['time'].values
    df_train['event'] = train_y['event'].values

    # Fit Cox model
    print(f"Fitting Cox PH model...")
    cox = CoxPHFitter(penalizer=0.1)
    cox.fit(df_train, duration_col='time', event_col='event')

    # Transform validation and test data
    X_val_pca = pca.transform(scaler.transform(val_X))
    X_test_pca = pca.transform(scaler.transform(test_X))

    # Evaluate
    train_risk = cox.predict_partial_hazard(pd.DataFrame(X_train_pca, columns=[f'PC{i}' for i in range(n_components)]))
    val_risk = cox.predict_partial_hazard(pd.DataFrame(X_val_pca, columns=[f'PC{i}' for i in range(n_components)]))
    test_risk = cox.predict_partial_hazard(pd.DataFrame(X_test_pca, columns=[f'PC{i}' for i in range(n_components)]))
    
    train_c_index = concordance_index(train_y['time'], -train_risk.values.ravel(), train_y['event'])
    val_c_index = concordance_index(val_y['time'], -val_risk.values.ravel(), val_y['event'])
    test_c_index = concordance_index(test_y['time'], -test_risk.values.ravel(), test_y['event'])

    print(f"Train C-index: {train_c_index:.4f}")
    print(f"Val C-index:   {val_c_index:.4f}")
    print(f"Test C-index:  {test_c_index:.4f}")

    # Get explained variance
    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"Explained variance: {explained_var:.1%}")

    results = {
        'model_name': f'cox_pca_{n_components}',
        'n_components': n_components,
        'train_c_index': float(train_c_index),
        'val_c_index': float(val_c_index),
        'test_c_index': float(test_c_index),
        'explained_variance': float(explained_var),
        'n_params': n_components,
    }

    return results


def train_cox_raw(train_data, val_data, test_data):
    """
    Train Cox PH on raw features (no dimensionality reduction).

    Tests: Can Cox handle high-dimensional data directly?
    """
    train_X, train_y = train_data
    val_X, val_y = val_data
    test_X, test_y = test_data

    print(f"\n{'='*80}")
    print(f"LINEAR BASELINE 2: Cox PH (Raw Features)")
    print(f"{'='*80}")

    print(f"Fitting Cox PH on {train_X.shape[1]} raw features...")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_X)
    
    df_train = pd.DataFrame(train_scaled)
    df_train['time'] = train_y['time'].values
    df_train['event'] = train_y['event'].values

    cox = CoxPHFitter(penalizer=1.0)
    cox.fit(df_train, duration_col='time', event_col='event')

    # Evaluate
    train_risk = cox.predict_partial_hazard(pd.DataFrame(scaler.transform(train_X)))
    val_risk = cox.predict_partial_hazard(pd.DataFrame(scaler.transform(val_X)))
    test_risk = cox.predict_partial_hazard(pd.DataFrame(scaler.transform(test_X)))
    
    train_c_index = concordance_index(train_y['time'], -train_risk.values.ravel(), train_y['event'])
    val_c_index = concordance_index(val_y['time'], -val_risk.values.ravel(), val_y['event'])
    test_c_index = concordance_index(test_y['time'], -test_risk.values.ravel(), test_y['event'])

    print(f"Train C-index: {train_c_index:.4f}")
    print(f"Val C-index:   {val_c_index:.4f}")
    print(f"Test C-index:  {test_c_index:.4f}")

    results = {
        'model_name': 'cox_raw',
        'n_features': train_X.shape[1],
        'train_c_index': float(train_c_index),
        'val_c_index': float(val_c_index),
        'test_c_index': float(test_c_index),
        'n_params': train_X.shape[1],
    }

    return results


def train_elastic_net_cox(train_data, val_data, test_data):
    """
    Train Elastic Net Cox (L1 + L2 regularization).

    Tests: Can sparse linear model with automatic feature selection work?
    """
    train_X, train_y = train_data
    val_X, val_y = val_data
    test_X, test_y = test_data

    print(f"\n{'='*80}")
    print(f"LINEAR BASELINE 3: Elastic Net Cox")
    print(f"{'='*80}")

    print(f"Fitting Elastic Net Cox (L1+L2 regularization)...")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_X)
    
    df_train = pd.DataFrame(train_scaled)
    df_train['time'] = train_y['time'].values
    df_train['event'] = train_y['event'].values

    cox = CoxPHFitter(penalizer=0.1, l1_ratio=0.5)
    cox.fit(df_train, duration_col='time', event_col='event')
    
    coef = cox.params_.values
    n_selected = np.sum(np.abs(coef) > 1e-5)

    # Evaluate
    train_risk = cox.predict_partial_hazard(pd.DataFrame(scaler.transform(train_X)))
    val_risk = cox.predict_partial_hazard(pd.DataFrame(scaler.transform(val_X)))
    test_risk = cox.predict_partial_hazard(pd.DataFrame(scaler.transform(test_X)))
    
    train_c_index = concordance_index(train_y['time'], -train_risk.values.ravel(), train_y['event'])
    val_c_index = concordance_index(val_y['time'], -val_risk.values.ravel(), val_y['event'])
    test_c_index = concordance_index(test_y['time'], -test_risk.values.ravel(), test_y['event'])

    print(f"Train C-index: {train_c_index:.4f}")
    print(f"Val C-index:   {val_c_index:.4f}")
    print(f"Test C-index:  {test_c_index:.4f}")
    print(f"Selected features: {n_selected}/{train_X.shape[1]}")

    results = {
        'model_name': 'elastic_net_cox',
        'n_features': train_X.shape[1],
        'n_selected': int(n_selected),
        'train_c_index': float(train_c_index),
        'val_c_index': float(val_c_index),
        'test_c_index': float(test_c_index),
        'n_params': int(n_selected),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Train linear baseline survival models")
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--prior_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/linear_baselines')
    args = parser.parse_args()

    # Set random seeds
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Linear Baseline Model Training")
    print("="*80)
    print(f"Output: {output_dir}")

    # Load data
    print("\n" + "="*80)
    print("Loading Data")
    print("="*80)

    graph_prior = load_graph_prior(args.prior_path)
    data_splits, survival_info, preprocessing_info = load_and_preprocess_survival_data(
        args.csv_path,
        graph_prior['protein_cols'],
        use_clinical=True,
        use_genomic=True
    )

    # Prepare data for Cox models
    train_data, val_data, test_data = prepare_survival_data(data_splits, survival_info)

    # Train all linear baselines
    all_results = {}

    # 1. Cox PH + PCA (multiple components)
    for n_comp in [10, 25, 50, 100]:
        results = train_cox_pca(train_data, val_data, test_data, n_components=n_comp)
        all_results[results['model_name']] = results

        # Save individual results
        results_path = output_dir / f"{results['model_name']}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

    # 2. Cox PH raw
    results = train_cox_raw(train_data, val_data, test_data)
    all_results[results['model_name']] = results

    results_path = output_dir / f"{results['model_name']}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # 3. Elastic Net Cox
    results = train_elastic_net_cox(train_data, val_data, test_data)
    all_results[results['model_name']] = results

    results_path = output_dir / f"{results['model_name']}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save summary
    summary_path = output_dir / 'linear_baseline_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print("Summary: Linear Baselines")
    print(f"{'='*80}")
    print(f"{'Model':<25} {'Test C-index':>12} {'Val C-index':>12} {'Params':>10}")
    print("-" * 80)

    for name, res in all_results.items():
        print(f"{name:<25} {res['test_c_index']:>12.4f} {res['val_c_index']:>12.4f} {res['n_params']:>10}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()