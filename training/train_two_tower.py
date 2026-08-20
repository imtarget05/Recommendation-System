"""Train and evaluate Two-Tower model on MovieLens 25M subset."""
from __future__ import annotations

import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, ".")

from training.two_tower import (
    TwoTowerDataset,
    TwoTowerModel,
    evaluate_retrieval,
    train_two_tower,
    two_tower_collate_fn,
)

random.seed(42)
np.random.seed(42)

print("Loading data...")
train_df = pd.read_parquet("data/processed/interactions_train.parquet")
test_df = pd.read_parquet("data/processed/interactions_test.parquet")

# Sample users that appear in BOTH train and test
test_users = test_df["user_id"].unique()
sampled = random.sample(list(test_users), 5000)
train_sub = train_df.loc[train_df["user_id"].isin(sampled)]
test_sub = test_df.loc[test_df["user_id"].isin(sampled)]
common = set(train_sub["user_id"]) & set(test_sub["user_id"])
train_sub = train_sub.loc[train_sub["user_id"].isin(list(common))]
test_sub = test_sub.loc[test_sub["user_id"].isin(list(common))]
print(f"Train: {train_sub.shape}, Test: {test_sub.shape}, users: {len(common)}")

items = sorted(set(train_sub["item_id"]) | set(test_sub["item_id"]))
users = sorted(common)
item_id_map = {iid: i for i, iid in enumerate(items)}
user_id_map = {uid: i for i, uid in enumerate(users)}

print(f"Building dataset ({len(users)} users, {len(items)} items)...")
dataset = TwoTowerDataset(train_sub, user_id_map, item_id_map, neg_samples=4)
loader = DataLoader(dataset, batch_size=256, shuffle=True, collate_fn=two_tower_collate_fn)

model = TwoTowerModel(n_users=len(user_id_map), n_items=len(item_id_map), emb_dim=64)

print("Training Two-Tower (3 epochs)...")
history = train_two_tower(model, loader, epochs=3, lr=0.001, device="cpu")
print(f"Training done. Final loss: {history['train_loss'][-1]:.4f}")

print("Evaluating...")
metrics = evaluate_retrieval(
    model, test_sub, user_id_map, item_id_map, train_sub,
    k_values=[10, 20], max_users=200,
)
print("Metrics:", metrics)

# Persist the checkpoint so the serving API (api/main.py, MODEL_PATH) can load it.
# The API loads keyed by 'model_state' with top-level n_users/n_items/emb_dim.
out_dir = os.environ.get("MODEL_OUTPUT_DIR", "outputs")
os.makedirs(out_dir, exist_ok=True)
model_path = os.path.join(out_dir, "two_tower.pt")
torch.save(
    {
        "model_state": model.state_dict(),
        "n_users": len(user_id_map),
        "n_items": len(item_id_map),
        "emb_dim": 64,
        "user_id_map": user_id_map,
        "item_id_map": item_id_map,
    },
    model_path,
)
print(f"Saved checkpoint to {model_path}")
