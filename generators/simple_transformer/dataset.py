import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from . import config


def normalize_stage(stage: str) -> str:
    if not isinstance(stage, str):
        return "UNKNOWN"
    stage = stage.strip().upper()
    if "I" in stage and "IV" not in stage and "III" not in stage and "II" not in stage:
        return "STAGE I"
    if "II" in stage and "III" not in stage:
        return "STAGE II"
    if "III" in stage:
        return "STAGE III"
    if "IV" in stage:
        return "STAGE IV"
    return "UNKNOWN"


def encode_sex(sex: str) -> str:
    if not isinstance(sex, str):
        return "UNKNOWN"
    sex = sex.strip().upper()
    if sex.startswith("F"):
        return "FEMALE"
    if sex.startswith("M"):
        return "MALE"
    return "UNKNOWN"


class SimpleRPPADataset(Dataset):
    def __init__(self, proteins, cancer_idx, stage_idx, sex_idx, age_norm):
        self.proteins = proteins
        self.cancer_idx = cancer_idx
        self.stage_idx = stage_idx
        self.sex_idx = sex_idx
        self.age_norm = age_norm

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.proteins[idx]).float(),
            {
                "cancer_type": torch.tensor(self.cancer_idx[idx], dtype=torch.long),
                "stage": torch.tensor(self.stage_idx[idx], dtype=torch.long),
                "sex": torch.tensor(self.sex_idx[idx], dtype=torch.long),
                "age": torch.tensor([self.age_norm[idx]], dtype=torch.float32),
            },
        )


def load_data() -> Tuple[Dict[str, Dataset], Dict]:
    cfg = config.DATA
    df = pd.read_csv(cfg["csv_path"])
    df = df[df["CANCER_TYPE_ACRONYM"].isin(cfg["selected_cancers"])].copy()

    df["stage_bucket"] = df["AJCC_PATHOLOGIC_TUMOR_STAGE"].map(normalize_stage)
    df["sex_clean"] = df["SEX"].map(encode_sex)
    df["age_clean"] = df["AGE"].astype(float)

    df = df.dropna(subset=["age_clean"])

    cancer_to_idx = {c: i for i, c in enumerate(cfg["selected_cancers"])}
    stage_to_idx = {s: i for i, s in enumerate(cfg["stage_buckets"])}
    sex_to_idx = {s: i for i, s in enumerate(cfg["sex_categories"])}

    cancer_idx = df["CANCER_TYPE_ACRONYM"].map(cancer_to_idx).to_numpy()
    stage_idx = df["stage_bucket"].map(lambda s: stage_to_idx.get(s, stage_to_idx["UNKNOWN"])).to_numpy()
    sex_idx = df["sex_clean"].map(lambda s: sex_to_idx.get(s, sex_to_idx["UNKNOWN"])).to_numpy()

    age_min, age_max = df["age_clean"].min(), df["age_clean"].max()
    age_norm = ((df["age_clean"] - age_min) / (age_max - age_min + 1e-8)).to_numpy()

    prior = np.load(cfg["prior_path"], allow_pickle=True)
    protein_cols = prior["protein_cols"].tolist()
    protein_df = df[protein_cols].copy()
    protein_df = protein_df.fillna(protein_df.mean())
    proteins = protein_df.to_numpy(dtype=np.float32)
    train_mask, temp_mask = train_test_split(
        np.arange(len(df)),
        test_size=(1 - cfg["train_ratio"]),
        random_state=cfg["seed"],
        stratify=cancer_idx,
    )
    val_size = cfg["val_ratio"] / (cfg["val_ratio"] + cfg["test_ratio"])
    val_mask, test_mask = train_test_split(
        temp_mask,
        test_size=1 - val_size,
        random_state=cfg["seed"],
        stratify=cancer_idx[temp_mask],
    )

    scaler = StandardScaler()
    proteins_train = scaler.fit_transform(proteins[train_mask])
    proteins_val = scaler.transform(proteins[val_mask])
    proteins_test = scaler.transform(proteins[test_mask])

    datasets = {
        "train": SimpleRPPADataset(
            proteins_train,
            cancer_idx[train_mask],
            stage_idx[train_mask],
            sex_idx[train_mask],
            age_norm[train_mask],
        ),
        "val": SimpleRPPADataset(
            proteins_val,
            cancer_idx[val_mask],
            stage_idx[val_mask],
            sex_idx[val_mask],
            age_norm[val_mask],
        ),
        "test": SimpleRPPADataset(
            proteins_test,
            cancer_idx[test_mask],
            stage_idx[test_mask],
            sex_idx[test_mask],
            age_norm[test_mask],
        ),
    }

    stage_probs = {}
    sex_probs = {}
    age_stats = {}
    for cancer_name, group in df.groupby("CANCER_TYPE_ACRONYM"):
        idx = cancer_to_idx[cancer_name]
        stage_counts = (
            group["stage_bucket"]
            .map(lambda s: stage_to_idx.get(s, stage_to_idx["UNKNOWN"]))
            .value_counts()
            .reindex(range(len(cfg["stage_buckets"])), fill_value=0)
        )
        sex_counts = (
            group["sex_clean"]
            .map(lambda s: sex_to_idx.get(s, sex_to_idx["UNKNOWN"]))
            .value_counts()
            .reindex(range(len(cfg["sex_categories"])), fill_value=0)
        )
        if stage_counts.sum() == 0:
            stage_probs[idx] = [1.0 / len(cfg["stage_buckets"])] * len(cfg["stage_buckets"])
        else:
            stage_probs[idx] = (stage_counts / stage_counts.sum()).tolist()

        if sex_counts.sum() == 0:
            sex_probs[idx] = [1.0 / len(cfg["sex_categories"])] * len(cfg["sex_categories"])
        else:
            sex_probs[idx] = (sex_counts / sex_counts.sum()).tolist()
        age_stats[idx] = {
            "mean": float(group["age_clean"].mean()),
            "std": float(max(group["age_clean"].std(), 1e-3)),
        }

    metadata = {
        "cancer_to_idx": cancer_to_idx,
        "idx_to_cancer": cfg["selected_cancers"],
        "stage_to_idx": stage_to_idx,
        "stage_buckets": cfg["stage_buckets"],
        "sex_to_idx": sex_to_idx,
        "sex_categories": cfg["sex_categories"],
        "age_min": float(age_min),
        "age_max": float(age_max),
        "protein_mean": scaler.mean_.tolist(),
        "protein_std": scaler.scale_.tolist(),
        "stage_probs": stage_probs,
        "sex_probs": sex_probs,
        "age_stats": age_stats,
    }

    return datasets, metadata, protein_cols


def create_dataloaders(datasets: Dict[str, Dataset]):
    dl = {}
    for split, ds in datasets.items():
        shuffle = split == "train"
        batch_size = config.TRAINING["batch_size"]
        dl[split] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return dl

