import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from simple_generator.dataset import load_data as load_base_data
from simple_generator.sample_and_classify import build_context, load_classifier

from . import config
from .graph_prior import load_graph_prior
from .model import SimpleTransformerGraphPrior

RESULTS_DIR = Path("graph_prior_transformer_samples_results")


def load_generator(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    metadata = ckpt["metadata"]
    protein_cols = ckpt["protein_cols"]

    prior_path = ckpt.get("graph_prior_path", config.GRAPH["prior_path"])
    graph_prior = load_graph_prior(prior_path)
    assert (
        graph_prior["protein_cols"] == protein_cols
    ), "Mismatch between checkpoint protein columns and graph prior."
    graph_tensors = {
        "PE": torch.from_numpy(graph_prior["PE"]).to(device),
        "K": torch.from_numpy(graph_prior["K"]).to(device),
    }

    model = SimpleTransformerGraphPrior(
        n_proteins=len(protein_cols),
        num_cancers=len(metadata["idx_to_cancer"]),
        num_stages=len(metadata["stage_buckets"]),
        num_sexes=len(metadata["sex_categories"]),
        graph_tensors=graph_tensors,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, metadata, protein_cols


def compute_train_counts(metadata):
    datasets, _, _ = load_base_data()
    cancer_names = metadata["idx_to_cancer"]
    train_counts = {}
    train_ds = datasets["train"]
    for idx, cancer in enumerate(cancer_names):
        train_counts[cancer] = int((train_ds.cancer_idx == idx).sum())
    return train_counts


def format_log(log_path, samples_per_class, cancer_names, accuracies, train_counts, mean_acc):
    random_baseline = 1.0 / len(cancer_names)
    with open(log_path, "w") as f:
        f.write("Graph-Prior Simple Generator Evaluation\n")
        f.write("----------------------------------------\n")
        f.write(f"Samples per cancer generated: {samples_per_class}\n")
        f.write(f"Number of cancers: {len(cancer_names)}\n")
        f.write(f"Random baseline accuracy: {random_baseline * 100:.2f}%\n")
        f.write(f"Mean accuracy: {mean_acc * 100:.3f}%\n\n")
        f.write("Per-cancer accuracy (%) and training counts\n")
        for cancer in cancer_names:
            acc = accuracies[cancer] * 100
            train_count = train_counts.get(cancer, 0)
            f.write(f"{cancer}: {acc:.2f}% (train samples: {train_count})\n")


def plot_accuracies(plot_path, cancer_names, accuracies, samples_per_class, train_counts):
    random_baseline = 1.0 / len(cancer_names)
    acc_values = [accuracies[c] * 100 for c in cancer_names]
    total_samples = samples_per_class * len(cancer_names)

    plt.figure(figsize=(12, 6))
    bars = plt.bar(
        cancer_names,
        acc_values,
        color="#5B8FF9",
        label=f"Graph-aware samples ({samples_per_class} per cancer, {total_samples} total)",
    )
    plt.axhline(
        random_baseline * 100,
        color="red",
        linestyle="--",
        label=f"Random baseline ({random_baseline*100:.2f}%)",
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Classifier accuracy (%)")
    plt.title("Graph Aware Transformer Samples Classification Accuracy")
    plt.ylim(-15, 105)
    for bar, cancer, val in zip(bars, cancer_names, acc_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            val + 1,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            -5,
            f"train n={train_counts.get(cancer, 0)}",
            ha="center",
            va="top",
            rotation=45,
            fontsize=7,
        )
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


def main():
    device = torch.device(config.TRAINING["device"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    generator_ckpt = Path(config.TRAINING["save_dir"]) / "best_simple_generator_graph_prior.pt"
    if not generator_ckpt.exists():
        raise FileNotFoundError(f"Generator checkpoint not found at {generator_ckpt}")

    model, metadata, protein_cols = load_generator(generator_ckpt, device)
    classifier, label_info = load_classifier(device)
    train_counts = compute_train_counts(metadata)

    mean = torch.tensor(metadata["protein_mean"], device=device)
    std = torch.tensor(metadata["protein_std"], device=device)
    cancer_names = metadata["idx_to_cancer"]
    samples_per_class = config.SAMPLING["samples_per_class"]

    results = {}
    per_cancer_correct = {}
    all_generated = []
    all_targets = []

    clf_names = label_info.get("cancer_types", label_info.get("class_names"))
    classifier_labels = clf_names if isinstance(clf_names, list) else list(clf_names)

    for cancer in cancer_names:
        context, targets = build_context(metadata, cancer, samples_per_class, device)
        with torch.no_grad():
            preds = model.generate(context)
        unnormalized = preds * std + mean
        all_generated.append(unnormalized.cpu())
        all_targets.append(targets.cpu())

        logits = classifier(unnormalized)
        pred_labels = logits.argmax(dim=-1)

        target_idx = classifier_labels.index(cancer)
        correct = (pred_labels == target_idx).float()
        accuracy = correct.mean().item()
        per_cancer_correct[cancer] = correct.sum().item()
        results[cancer] = accuracy
        print(f"{cancer}: classifier accuracy on generated = {accuracy:.3f}")

        np.save(RESULTS_DIR / f"{cancer}_samples.npy", unnormalized.cpu().numpy())

    generated = torch.cat(all_generated, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    np.save(RESULTS_DIR / "all_generated.npy", generated)
    np.save(RESULTS_DIR / "all_targets.npy", targets)

    total_correct = sum(per_cancer_correct.values())
    total_samples = samples_per_class * len(cancer_names)
    mean_accuracy = total_correct / total_samples if total_samples else 0.0

    metrics_path = RESULTS_DIR / "classifier_accuracy.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "per_cancer_accuracy": results,
                "mean_accuracy": mean_accuracy,
                "random_baseline": 1.0 / len(cancer_names),
                "samples_per_class": samples_per_class,
            },
            f,
            indent=2,
        )
    print(f"Saved results to {metrics_path}")

    log_path = RESULTS_DIR / "classifier_accuracy_log.txt"
    format_log(log_path, samples_per_class, cancer_names, results, train_counts, mean_accuracy)

    plot_path = RESULTS_DIR / "classifier_accuracy_vs_random.png"
    plot_accuracies(plot_path, cancer_names, results, samples_per_class, train_counts)
    print(f"Wrote log to {log_path} and plot to {plot_path}")


if __name__ == "__main__":
    main()

