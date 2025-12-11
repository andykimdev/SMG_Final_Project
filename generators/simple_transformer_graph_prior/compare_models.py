import argparse
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from simple_generator.dataset import load_data, create_dataloaders
from simple_generator.model import SimpleTransformerGenerator

from . import config
from .graph_prior import load_graph_prior
from .model import SimpleTransformerGraphPrior


def to_device(context, device):
    return {k: v.to(device) for k, v in context.items()}


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for proteins, context in loader:
            proteins = proteins.to(device)
            context = to_device(context, device)
            preds = model(context)
            loss = loss_fn(preds, proteins)
            total_loss += loss.item() * proteins.size(0)
    return total_loss / len(loader.dataset)


def load_simple_model(ckpt_path, metadata, protein_cols, device):
    model = SimpleTransformerGenerator(
        n_proteins=len(protein_cols),
        num_cancers=len(metadata["idx_to_cancer"]),
        num_stages=len(metadata["stage_buckets"]),
        num_sexes=len(metadata["sex_categories"]),
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    return model


def load_graph_prior_model(ckpt_path, metadata, protein_cols, prior_path, device):
    state = torch.load(ckpt_path, map_location=device)
    graph_prior = load_graph_prior(prior_path)
    assert (
        graph_prior["protein_cols"] == protein_cols
    ), "Protein columns mismatch between checkpoint and prior."
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
    model.load_state_dict(state["model_state"])
    return model


def main():
    parser = argparse.ArgumentParser(description="Compare simple and graph-prior transformers.")
    parser.add_argument(
        "--simple_ckpt",
        type=str,
        default="outputs/best_simple_generator.pt",
        help="Path to baseline simple transformer checkpoint.",
    )
    parser.add_argument(
        "--graph_ckpt",
        type=str,
        default="outputs_graph_prior/best_simple_generator_graph_prior.pt",
        help="Path to graph-prior transformer checkpoint.",
    )
    parser.add_argument(
        "--prior_path",
        type=str,
        default=config.GRAPH["prior_path"],
        help="STRING prior path (needed for graph model).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split for evaluation.",
    )
    args = parser.parse_args()

    torch.manual_seed(config.DATA["seed"])
    datasets, metadata, protein_cols = load_data()
    loaders = create_dataloaders(datasets)

    device = torch.device(config.TRAINING["device"])
    loss_fn = nn.MSELoss()

    simple_model = load_simple_model(args.simple_ckpt, metadata, protein_cols, device)
    graph_model = load_graph_prior_model(
        args.graph_ckpt, metadata, protein_cols, args.prior_path, device
    )

    loader = loaders[args.split]
    print(f"Evaluating on {args.split} split ({len(loader.dataset)} samples)")
    simple_loss = evaluate(simple_model, loader, loss_fn, device)
    graph_loss = evaluate(graph_model, loader, loss_fn, device)

    print(f"Simple transformer loss:     {simple_loss:.4f}")
    print(f"Graph-prior transformer loss:{graph_loss:.4f}")


if __name__ == "__main__":
    main()

