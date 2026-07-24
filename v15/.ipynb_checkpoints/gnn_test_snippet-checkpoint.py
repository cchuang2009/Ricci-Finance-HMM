"""
Standalone GNN test snippet for Ricci Finance V15.1.

Run this only after the notebook or your own pipeline has created:
    frames
    feature_df
    hmm_states

No torch_geometric, torch_scatter, torch_sparse, or torch_xla is used.
"""

import os

# Avoid native BLAS/OpenMP thread conflicts.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Test CPU first.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())




import inspect
import numpy as np
import pandas as pd

from ricci_finance.gnn import train_gcn_regime
from ricci_finance.sectors import assign_sectors

RANDOM_STATE = 42


#0. Load Frame
import pickle

with open("ricci_dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

frames = dataset["frames"]
feature_df = dataset["feature_df"]
hmm_states = dataset["hmm_states"]
regime_names = dataset["regime_names"]

# 1. Build aligned inputs.
graph_list = [frame["graph"] for frame in frames]
labels = np.asarray(hmm_states, dtype=int)

n_samples = min(len(graph_list), len(labels), len(feature_df))
graph_list = graph_list[:n_samples]
labels = labels[:n_samples]
dates = pd.to_datetime(feature_df["date"].iloc[:n_samples]).to_numpy()

if n_samples < 8:
    raise ValueError(
        f"Only {n_samples} samples are available. "
        "Increase PERIOD or reduce WINDOW/STEP."
    )

all_nodes = sorted({
    node
    for graph in graph_list
    for node in graph.nodes()
})
sectors = assign_sectors(all_nodes)

print("Samples:", n_samples)
print("Nodes:", len(all_nodes))
print("Class counts:", dict(zip(*np.unique(labels, return_counts=True))))

# 2. Use only parameters supported by the installed gnn.py revision.
candidate_kwargs = {
    "epochs": 250,
    "hidden": 32,
    "dropout": 0.25,
    "learning_rate": 0.003,
    "weight_decay": 1e-4,
    "train_fraction": 0.70,
    "patience": 40,
    "random_state": RANDOM_STATE,
    "use_cuda": True,
    "predict_next_state": False,
}

signature = inspect.signature(train_gcn_regime)
supported_kwargs = {
    key: value
    for key, value in candidate_kwargs.items()
    if key in signature.parameters
}

# 3. Train.
result = train_gcn_regime(
    graph_list,
    labels,
    sectors,
    **supported_kwargs,
)

print(result.note)
print("Accuracy:", result.accuracy)
print(
    "Balanced accuracy:",
    getattr(result, "balanced_accuracy", "not reported"),
)
print("Device:", getattr(result, "device", "not reported"))
print("Epochs:", getattr(result, "epochs", "not reported"))

# 4. Build a comparison table compatible with old and new revisions.
predictions = np.asarray(result.predictions)
test_indices = getattr(result, "test_indices", None)

if test_indices is not None:
    test_indices = np.asarray(test_indices, dtype=int)
    comparison = pd.DataFrame({
        "date": dates[test_indices],
        "HMM": labels[test_indices],
        "GCN": predictions[test_indices],
    })

    probabilities = getattr(result, "probabilities", None)
    if probabilities is not None:
        probabilities = np.asarray(probabilities)
        for class_index in range(probabilities.shape[1]):
            comparison[f"P_class_{class_index}"] = (
                probabilities[test_indices, class_index]
            )
else:
    comparison = pd.DataFrame({
        "date": dates[-len(predictions):],
        "HMM": labels[-len(predictions):],
        "GCN": predictions,
    })

print("\nPredictions")
print(comparison.to_string(index=False))

print("\nConfusion matrix")
print(pd.crosstab(
    comparison["HMM"],
    comparison["GCN"],
    rownames=["Actual HMM"],
    colnames=["Predicted GCN"],
    dropna=False,
))
