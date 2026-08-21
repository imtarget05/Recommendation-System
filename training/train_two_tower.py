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
items_df = pd.read_parquet("data/processed/items.parquet")

# Build id maps EXACTLY like api/main.py reconstructs them at serving time
# (items from items.parquet in file order, users from the train split). This
# guarantees the saved embedding-table shapes match what the serving container
# rebuilds, so the checkpoint loads cleanly instead of falling back to
# "untrained".
item_id_map = {iid: i for i, iid in enumerate(items_df["item_id"].tolist())}
user_id_map = {uid: i for i, uid in enumerate(sorted(train_df["user_id"].unique()))}
n_items = len(item_id_map)
n_users = len(user_id_map)
print(f"Train: {train_df.shape}, users={n_users}, items={n_items}")

print(f"Building dataset ({n_users} users, {n_items} items)...")
dataset = TwoTowerDataset(train_df, user_id_map, item_id_map, neg_samples=4)
loader = DataLoader(dataset, batch_size=256, shuffle=True, collate_fn=two_tower_collate_fn)

model = TwoTowerModel(n_users=n_users, n_items=n_items, emb_dim=64)

print("Training Two-Tower (3 epochs)...")
history = train_two_tower(model, loader, epochs=3, lr=0.001, device="cpu")
print(f"Training done. Final loss: {history['train_loss'][-1]:.4f}")

print("Evaluating...")
# Evaluate on users present in BOTH train and test (capped to keep it tractable).
common = sorted(set(train_df["user_id"]) & set(test_df["user_id"]))
eval_users = random.sample(common, min(2000, len(common)))
test_sub = test_df.loc[test_df["user_id"].isin(eval_users)]
metrics = evaluate_retrieval(
    model, test_sub, user_id_map, item_id_map, train_df,
    k_values=[10, 20], max_users=2000,
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
        "version": os.environ.get("MODEL_VERSION", "two-tower-v1"),
        "user_id_map": user_id_map,
        "item_id_map": item_id_map,
    },
    model_path,
)
print(f"Saved checkpoint to {model_path}")
