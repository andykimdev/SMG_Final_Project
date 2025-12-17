import json
from pathlib import Path

import numpy as np
import torch

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root / "Cancer Classifier"))
from graph_transformer_classifier import GraphTransformerClassifier
from graph_prior import load_graph_prior

from . import config
from .model import SimpleTransformerGenerator


def load_generator(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    metadata = ckpt["metadata"]
    protein_cols = ckpt["protein_cols"]
    model = SimpleTransformerGenerator(
        n_proteins=len(protein_cols),
        num_cancers=len(metadata["idx_to_cancer"]),
        num_stages=len(metadata["stage_buckets"]),
        num_sexes=len(metadata["sex_categories"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, metadata, protein_cols


def build_context(metadata, cancer_name, num_samples, device):
    cancer_to_idx = metadata["cancer_to_idx"]
    idx = cancer_to_idx[cancer_name]
    stage_probs = metadata["stage_probs"][idx]
    sex_probs = metadata["sex_probs"][idx]
    age_stat = metadata["age_stats"][idx]

    stage_choices = np.random.choice(
        len(stage_probs), size=num_samples, p=np.array(stage_probs)
    )
    sex_choices = np.random.choice(
        len(sex_probs), size=num_samples, p=np.array(sex_probs)
    )
    ages = np.random.normal(age_stat["mean"], age_stat["std"], size=num_samples)
    ages = np.clip(ages, metadata["age_min"], metadata["age_max"])
    ages_norm = (ages - metadata["age_min"]) / (metadata["age_max"] - metadata["age_min"] + 1e-8)

    context = {
        "cancer_type": torch.full((num_samples,), idx, dtype=torch.long, device=device),
        "stage": torch.tensor(stage_choices, dtype=torch.long, device=device),
        "sex": torch.tensor(sex_choices, dtype=torch.long, device=device),
        "age": torch.tensor(ages_norm, dtype=torch.float32, device=device).unsqueeze(-1),
    }
    targets = torch.tensor([idx] * num_samples, dtype=torch.long, device=device)
    return context, targets


def load_classifier(device):
    checkpoint = torch.load(
        "Cancer Classifier/outputs/checkpoints/best_model.pt",
        map_location=device,
    )
    label_info = checkpoint["label_info"]
    graph_prior = load_graph_prior(str(repo_root / "priors" / "tcga_string_prior.npz"))
    K = torch.from_numpy(graph_prior["K"]).float().to(device)
    PE = torch.from_numpy(graph_prior["PE"]).float().to(device)

    clf_config = checkpoint.get("config", {}).get("MODEL", {})
    classifier = GraphTransformerClassifier(
        n_proteins=K.shape[0],
        n_classes=label_info["n_classes"],
        diffusion_kernel=K,
        positional_encodings=PE,
        embedding_dim=clf_config.get("embedding_dim"),
        n_layers=clf_config.get("n_layers"),
        n_heads=clf_config.get("n_heads"),
        ffn_dim=clf_config.get("ffn_dim"),
        dropout=clf_config.get("dropout"),
    ).to(device)
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.eval()
    return classifier, label_info


def main():
    device = torch.device(config.TRAINING["device"])
    save_dir = Path(config.SAMPLING["output_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    generator_ckpt = Path(config.TRAINING["save_dir"]) / "best_simple_generator.pt"
    if not generator_ckpt.exists():
        raise FileNotFoundError(f"Generator checkpoint not found at {generator_ckpt}")

    model, metadata, protein_cols = load_generator(generator_ckpt, device)
    classifier, label_info = load_classifier(device)

    mean = torch.tensor(metadata["protein_mean"], device=device)
    std = torch.tensor(metadata["protein_std"], device=device)

    cancer_names = metadata["idx_to_cancer"]
    samples_per_class = config.SAMPLING["samples_per_class"]

    results = {}
    all_generated = []
    all_targets = []
    for cancer in cancer_names:
        context, targets = build_context(metadata, cancer, samples_per_class, device)
        with torch.no_grad():
            preds = model.generate(context)
        unnormalized = preds * std + mean
        all_generated.append(unnormalized.cpu())
        all_targets.append(targets.cpu())

        logits = classifier(unnormalized)
        pred_labels = logits.argmax(dim=-1)

        # Map classifier labels to names
        clf_names = label_info.get("cancer_types", label_info.get("class_names"))
        clf_name_list = clf_names if isinstance(clf_names, list) else list(clf_names)
        target_name = cancer
        target_label_idx = clf_name_list.index(target_name)
        accuracy = (pred_labels == target_label_idx).float().mean().item()
        results[cancer] = accuracy
        print(f"{cancer}: classifier accuracy on generated = {accuracy:.3f}")

        np.save(save_dir / f"{cancer}_samples.npy", unnormalized.cpu().numpy())

    generated = torch.cat(all_generated, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    np.save(save_dir / "all_generated.npy", generated)
    np.save(save_dir / "all_targets.npy", targets)

    metrics_path = save_dir / "classifier_accuracy.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {metrics_path}")


if __name__ == "__main__":
    main()

