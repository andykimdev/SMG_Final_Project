import json
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from . import config
from .dataset import create_dataloaders, load_data
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


def build_graph_tensors(graph_prior, device):
    return {
        "PE": torch.from_numpy(graph_prior["PE"]).to(device),
        "K": torch.from_numpy(graph_prior["K"]).to(device),
    }


def main():
    torch.manual_seed(config.DATA["seed"])
    datasets, metadata, protein_cols = load_data()
    loaders = create_dataloaders(datasets)

    device = torch.device(config.TRAINING["device"])
    graph_prior = load_graph_prior(config.GRAPH["prior_path"])
    assert (
        graph_prior["protein_cols"] == protein_cols
    ), "Protein columns from dataset/prior are misaligned."
    graph_tensors = build_graph_tensors(graph_prior, device)

    model = SimpleTransformerGraphPrior(
        n_proteins=len(protein_cols),
        num_cancers=len(metadata["idx_to_cancer"]),
        num_stages=len(metadata["stage_buckets"]),
        num_sexes=len(metadata["sex_categories"]),
        graph_tensors=graph_tensors,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.TRAINING["learning_rate"],
        weight_decay=config.TRAINING["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    loss_fn = nn.MSELoss()

    save_dir = Path(config.TRAINING["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = save_dir / "best_simple_generator_graph_prior.pt"

    history = {"epoch": [], "train_loss": [], "val_loss": [], "lr": []}
    history_path = Path(config.LOGGING["history_file"])
    history_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    patience = 0

    for epoch in range(1, config.TRAINING["max_epochs"] + 1):
        model.train()
        train_loss = 0.0
        loop = tqdm(
            loaders["train"],
            desc=f"Epoch {epoch}/{config.TRAINING['max_epochs']}",
            leave=False,
        )
        for proteins, context in loop:
            proteins = proteins.to(device)
            context = to_device(context, device)
            preds = model(context)
            loss = loss_fn(preds, proteins)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * proteins.size(0)
            loop.set_postfix(loss=loss.item())
        train_loss /= len(loaders["train"].dataset)

        val_loss = evaluate(model, loaders["val"], loss_fn, device)
        scheduler.step(val_loss)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss + config.TRAINING["min_delta"] < best_val:
            best_val = val_loss
            patience = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "metadata": metadata,
                    "protein_cols": protein_cols,
                    "config": config.MODEL,
                    "graph_config": config.GRAPH,
                    "graph_prior_path": config.GRAPH["prior_path"],
                },
                best_ckpt,
            )
            print(f"  ✓ New best model saved to {best_ckpt}")
        else:
            patience += 1
            if patience >= config.TRAINING["patience"]:
                print("Early stopping triggered.")
                break

    with open(history_path, "w") as fp:
        json.dump(history, fp, indent=2)
    print(f"Saved training history to {history_path}")
    print("Training complete.")


if __name__ == "__main__":
    main()

