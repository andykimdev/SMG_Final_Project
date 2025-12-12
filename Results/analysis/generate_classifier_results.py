#!/usr/bin/env python3
"""
Generate unified classifier evaluation artifacts for generator variants.

For each model produces:
- Per-cancer accuracy JSON + log + bar chart (with training sample counts)
- Classifier prediction distribution JSON + bar chart
- Text log summarizing counts, baselines, and accuracy numbers
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]


def add_sys_paths() -> None:
    """Ensure project modules are importable."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
    sys.path.append(str(REPO_ROOT / "classifiers" / "graph_transformer"))
    sys.path.append(str(REPO_ROOT / "generators" / "diffusion"))
    sys.path.append(str(REPO_ROOT / "generators" / "simple_transformer"))
    sys.path.append(str(REPO_ROOT / "generators" / "simple_transformer_graph_prior"))


add_sys_paths()

from classifiers.graph_transformer.graph_prior import load_graph_prior
from classifiers.graph_transformer.model import GraphTransformerClassifier

from generators.simple_transformer import config as simple_config
from generators.simple_transformer import dataset as simple_dataset


def load_module_from_path(path: Path, module_name: str):
    """Dynamically load a module when it lacks a package __init__."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DATASET_DIFFUSION = load_module_from_path(
    REPO_ROOT / "generators" / "diffusion" / "dataset_diffusion.py",
    "dataset_diffusion_runtime",
)

if "val_ratio" not in DATASET_DIFFUSION.config.DATA:
    DATASET_DIFFUSION.config.DATA["val_ratio"] = DATASET_DIFFUSION.config.VAL_RATIO
if "test_ratio" not in DATASET_DIFFUSION.config.DATA:
    DATASET_DIFFUSION.config.DATA["test_ratio"] = DATASET_DIFFUSION.config.TEST_RATIO
if "train_ratio" not in DATASET_DIFFUSION.config.DATA:
    DATASET_DIFFUSION.config.DATA["train_ratio"] = DATASET_DIFFUSION.config.TRAIN_RATIO


def build_classifier(device: torch.device) -> Tuple[GraphTransformerClassifier, List[str]]:
    """Load the trained classifier and graph priors."""
    ckpt_path = REPO_ROOT / "Results" / "classifiers" / "cancer_type_classifiers" / "transformer" / "checkpoints" / "best_model.pt"
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    graph_prior = load_graph_prior(str(REPO_ROOT / "priors" / "tcga_string_prior.npz"))
    K = torch.from_numpy(graph_prior["K"]).float().to(device)
    PE = torch.from_numpy(graph_prior["PE"]).float().to(device)

    clf_cfg = checkpoint["config"]["MODEL"]
    model = GraphTransformerClassifier(
        n_proteins=K.shape[0],
        n_classes=checkpoint["label_info"]["n_classes"],
        diffusion_kernel=K,
        positional_encodings=PE,
        embedding_dim=clf_cfg.get("embedding_dim"),
        n_layers=clf_cfg.get("n_layers"),
        n_heads=clf_cfg.get("n_heads"),
        ffn_dim=clf_cfg.get("ffn_dim"),
        dropout=clf_cfg.get("dropout"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    class_names = checkpoint["label_info"]["class_names"]
    return model, class_names


def compute_simple_train_counts() -> Dict[str, int]:
    """Count training samples per cancer type for the simple generators."""
    datasets, metadata, _ = simple_dataset.load_data()
    train_ds = datasets["train"]
    counts = {}
    for cancer, idx in metadata["cancer_to_idx"].items():
        mask = train_ds.cancer_idx == idx
        counts[cancer] = int(mask.sum())
    return counts


def compute_diffusion_train_counts() -> Dict[str, int]:
    """Count training samples per cancer type for the diffusion dataset."""
    prior = np.load(REPO_ROOT / "priors" / "tcga_string_prior.npz", allow_pickle=True)
    protein_cols = prior["protein_cols"].tolist()
    csv_path = REPO_ROOT / "processed_datasets" / "tcga_pancan_rppa_compiled.csv"

    data_splits, _, _, label_info = DATASET_DIFFUSION.load_and_preprocess_diffusion_data(
        str(csv_path),
        protein_cols=protein_cols,
    )
    train_ds = data_splits["train"]
    counts = Counter()
    cancer_types = train_ds.cancer_types.numpy()
    for idx in cancer_types:
        counts[label_info["cancer_types"][int(idx)]] += 1
    return dict(counts)


def load_samples_from_dir(sample_dir: Path) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    """Load stacked samples and target labels from per-cancer .npy files."""
    tensors = []
    true_labels: List[str] = []
    per_cancer_counts: Dict[str, int] = {}
    for path in sorted(sample_dir.glob("*_samples.npy")):
        cancer = path.stem.replace("_samples", "")
        arr = np.load(path)
        arr = arr.astype(np.float32)
        tensors.append(arr)
        true_labels.extend([cancer] * arr.shape[0])
        per_cancer_counts[cancer] = per_cancer_counts.get(cancer, 0) + arr.shape[0]
    if not tensors:
        raise FileNotFoundError(f"No *_samples.npy files found in {sample_dir}")
    stacked = np.concatenate(tensors, axis=0)
    return stacked, true_labels, per_cancer_counts


def run_classifier(model, samples: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    preds = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            end = start + batch_size
            batch = torch.from_numpy(samples[start:end]).float().to(device)
            logits = model(batch)
            preds.append(logits.argmax(dim=-1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def compute_accuracy(true_labels: List[str], pred_labels: List[str]) -> Tuple[Dict[str, float], float]:
    total = len(true_labels)
    correct = sum(int(t == p) for t, p in zip(true_labels, pred_labels))
    overall = correct / total if total > 0 else 0.0

    per_class = OrderedDict()
    counts = Counter(true_labels)
    correct_counts = Counter()
    for t, p in zip(true_labels, pred_labels):
        if t == p:
            correct_counts[t] += 1
    for cls in counts:
        per_class[cls] = correct_counts[cls] / counts[cls] if counts[cls] else 0.0
    return per_class, overall


def plot_accuracy(
    classes: List[str],
    accuracies: Dict[str, float],
    train_counts: Dict[str, int],
    random_baseline: float,
    title: str,
    legend_note: str,
    output_path: Path,
) -> None:
    values = [accuracies.get(c, 0.0) * 100 for c in classes]
    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 0.6), 4.5))
    bars = ax.bar(classes, values, color="#4e79a7")
    ax.axhline(random_baseline, color="#f28e2b", linestyle="--", linewidth=2, label=f"Random baseline ({random_baseline:.1f}%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.set_ylim(0, max(values + [random_baseline]) + 15)
    plt.xticks(rotation=35, ha="right")
    for bar, cls in zip(bars, classes):
        train_count = train_counts.get(cls, 0)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{train_count} train",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.legend(title=legend_note)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_prediction_distribution(
    distribution: Dict[str, float],
    title: str,
    output_path: Path,
) -> None:
    sorted_items = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
    classes = [c for c, _ in sorted_items]
    values = [v for _, v in sorted_items]
    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 0.45), 4.0))
    ax.bar(classes, values, color="#59a14f")
    ax.set_ylabel("Prediction share (%)")
    ax.set_title(title)
    ax.set_ylim(0, max(values + [1.0]) + 5)
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_log(
    path: Path,
    model_name: str,
    overall_acc: float,
    random_baseline: float,
    target_counts: Dict[str, int],
    accuracies: Dict[str, float],
    train_counts: Dict[str, int],
    pred_distribution: Dict[str, float],
) -> None:
    lines = [
        f"{model_name}",
        "-" * len(model_name),
        f"Total generated samples: {sum(target_counts.values())}",
        f"Unique cancer types: {len(target_counts)}",
        f"Overall classifier accuracy: {overall_acc * 100:.2f}%",
        f"Random baseline: {random_baseline:.2f}%",
        "",
        "Per-cancer accuracy (train count → accuracy, generated samples):",
    ]
    for cancer in target_counts:
        lines.append(
            f"  {cancer:>6}: train={train_counts.get(cancer, 0):4d}, "
            f"generated={target_counts[cancer]:3d}, acc={accuracies.get(cancer, 0.0) * 100:5.2f}%"
        )
    lines.extend(
        [
            "",
            "Classifier prediction distribution (% of total predictions):",
        ]
    )
    for cancer, pct in sorted(pred_distribution.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  {cancer:>6}: {pct:6.2f}%")
    path.write_text("\n".join(lines) + "\n")


def process_model(
    model_key: str,
    display_name: str,
    sample_source: str,
    source_path: Path,
    result_dir: Path,
    class_order: Optional[List[str]],
    train_counts: Dict[str, int],
    classifier: GraphTransformerClassifier,
    class_names: List[str],
    device: torch.device,
    legend_note: str,
) -> Dict[str, str]:
    ensure_dir(result_dir)
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    if sample_source == "dir":
        samples, true_labels, target_counts = load_samples_from_dir(source_path)
        pred_indices = run_classifier(classifier, samples, batch_size=256, device=device)
        pred_labels = [class_names[i] for i in pred_indices]
    elif sample_source == "npz":
        data = np.load(source_path, allow_pickle=True)
        samples = data["proteins"].astype(np.float32)
        true_label_arr = data["true_labels"]
        true_labels = [str(x) for x in true_label_arr]
        target_counts = Counter(true_labels)
        pred_indices = run_classifier(classifier, samples, batch_size=256, device=device)
        pred_labels = [class_names[i] for i in pred_indices]
    else:
        raise ValueError(f"Unknown sample_source: {sample_source}")

    accuracies, overall_acc = compute_accuracy(true_labels, pred_labels)
    pred_counts = Counter(pred_labels)
    total_preds = sum(pred_counts.values())
    pred_distribution = {cls: (pred_counts.get(cls, 0) / total_preds) * 100 for cls in pred_counts}

    if class_order is None:
        ordered_classes = [c for c in class_names if c in target_counts]
    else:
        ordered_classes = [c for c in class_order if c in target_counts]

    if not ordered_classes:
        ordered_classes = list(target_counts.keys())

    random_baseline = 100.0 / max(len(ordered_classes), 1)

    accuracy_plot = result_dir / "classifier_accuracy_vs_random.png"
    plot_accuracy(
        ordered_classes,
        accuracies,
        train_counts,
        random_baseline,
        f"{display_name} → Classifier accuracy",
        legend_note,
        accuracy_plot,
    )

    distribution_plot = result_dir / "classifier_prediction_distribution.png"
    plot_prediction_distribution(
        pred_distribution,
        f"{display_name} → Classifier prediction mix",
        distribution_plot,
    )

    accuracy_json = result_dir / "classifier_accuracy.json"
    accuracy_json.write_text(json.dumps({cls: accuracies.get(cls, 0.0) for cls in ordered_classes}, indent=2))

    distribution_json = result_dir / "prediction_distribution.json"
    distribution_json.write_text(json.dumps(pred_distribution, indent=2))

    log_path = result_dir / "classifier_accuracy_log.txt"
    write_log(
        log_path,
        display_name,
        overall_acc,
        random_baseline,
        dict(target_counts),
        accuracies,
        train_counts,
        pred_distribution,
    )

    return {
        "key": model_key,
        "name": display_name,
        "overall_acc": f"{overall_acc * 100:.2f}%",
        "log": str(log_path.relative_to(REPO_ROOT)),
        "accuracy_plot": str(accuracy_plot.relative_to(REPO_ROOT)),
        "distribution_plot": str(distribution_plot.relative_to(REPO_ROOT)),
    }


def write_summary_file(summaries: List[Dict[str, str]]) -> None:
    summary_dir = REPO_ROOT / "Results" / "analysis"
    ensure_dir(summary_dir)
    lines = ["# Classifier Evaluation Summary", ""]
    for item in summaries:
        lines.extend(
            [
                f"## {item['name']}",
                f"- Overall accuracy on generated samples: {item['overall_acc']}",
                f"- Log: `{item['log']}`",
                f"- Accuracy plot: `{item['accuracy_plot']}`",
                f"- Prediction distribution: `{item['distribution_plot']}`",
                "",
            ]
        )
    (summary_dir / "summary.md").write_text("\n".join(lines).rstrip() + "\n")


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    classifier, class_names = build_classifier(device)

    print("Computing training counts...")
    simple_counts = compute_simple_train_counts()
    diffusion_counts = compute_diffusion_train_counts()

    models = [
        {
            "key": "simple_transformer",
            "display": "Simple Transformer Generator",
            "sample_source": "dir",
            "source_path": REPO_ROOT / "Results" / "generators" / "simple_transformer" / "generated",
            "result_dir": REPO_ROOT / "Results" / "generators" / "simple_transformer",
            "class_order": simple_config.DATA["selected_cancers"],
            "train_counts": simple_counts,
            "legend": "50 generator samples per cancer",
        },
        {
            "key": "graph_prior",
            "display": "Graph-Prior Transformer Generator",
            "sample_source": "dir",
            "source_path": REPO_ROOT / "Results" / "generators" / "simple_transformer_graph_prior" / "generated",
            "result_dir": REPO_ROOT / "Results" / "generators" / "simple_transformer_graph_prior",
            "class_order": simple_config.DATA["selected_cancers"],
            "train_counts": simple_counts,
            "legend": "50 generator samples per cancer",
        },
        {
            "key": "diffusion",
            "display": "Graph Diffusion Generator",
            "sample_source": "npz",
            "source_path": REPO_ROOT / "Results" / "generators" / "diffusion" / "validation_results" / "generated_samples.npz",
            "result_dir": REPO_ROOT / "Results" / "generators" / "diffusion",
            "class_order": None,
            "train_counts": diffusion_counts,
            "legend": "1000 synthetic contexts",
        },
    ]

    summaries = []
    for model in models:
        print(f"Processing {model['display']}...")
        summary = process_model(
            model_key=model["key"],
            display_name=model["display"],
            sample_source=model["sample_source"],
            source_path=model["source_path"],
            result_dir=model["result_dir"],
            class_order=model["class_order"],
            train_counts=model["train_counts"],
            classifier=classifier,
            class_names=class_names,
            device=device,
            legend_note=model["legend"],
        )
        summaries.append(summary)

    write_summary_file(summaries)
    print("Classifier result artifacts updated.")


if __name__ == "__main__":
    main()

