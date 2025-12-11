import json
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from . import config
from .dataset import load_data, create_dataloaders
from .model import SimpleTransformerGenerator


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


def main():
    torch.manual_seed(config.DATA["seed"])
    datasets, metadata, protein_cols = load_data()
    loaders = create_dataloaders(datasets)

    device = torch.device(config.TRAINING["device"])
    model = SimpleTransformerGenerator(
        n_proteins=len(protein_cols),
        num_cancers=len(metadata["idx_to_cancer"]),
        num_stages=len(metadata["stage_buckets"]),
        num_sexes=len(metadata["sex_categories"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.TRAINING["learning_rate"],
        weight_decay=config.TRAINING["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5
    )
    loss_fn = nn.MSELoss()

    save_dir = Path(config.TRAINING["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = save_dir / "best_simple_generator.pt"

    best_val = float("inf")
    patience = 0

    for epoch in range(1, config.TRAINING["max_epochs"] + 1):
        model.train()
        train_loss = 0.0
        loop = tqdm(loaders["train"], desc=f"Epoch {epoch}/{config.TRAINING['max_epochs']}", leave=False)
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
                },
                best_ckpt,
            )
            print(f"  ✓ New best model saved to {best_ckpt}")
        else:
            patience += 1
            if patience >= config.TRAINING["patience"]:
                print("Early stopping triggered.")
                break

    print("Training complete.")


if __name__ == "__main__":
    main()

