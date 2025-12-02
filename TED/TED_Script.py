#!/usr/bin/env python3
"""
TED (Topological Evolution Dynamics) – Minimal runnable script
================================================================
This script implements the core idea in the paper
"Robust Backdoor Detection for Deep Learning via Topological Evolution Dynamics"
(aka TED detector) in a self‑contained way using PyTorch + torchvision.

It:
  1) Loads a pretrained classifier and a dataset (MNIST / CIFAR‑10).
  2) Hooks all Conv2d layers to collect per‑layer activations.
  3) Builds a small per‑class clean support set (m samples per class).
  4) For any input sample, computes its **rank feature vector** K = [K_1..K_L]
     where K_l is the rank position of the nearest neighbor from the **predicted
     class** among all support samples at layer l (distance in Euclidean metric).
  5) Fits a simple PCA‑based outlier detector on the rank features of the clean
     support set (kept outliers proportion = alpha). The anomaly score is the
     reconstruction error in the PCA subspace; threshold is the (1‑alpha) quantile.
  6) Scores a test split and writes results to CSV.

Notes:
 - This is a minimal, model‑agnostic implementation faithful to the paper’s
   Algorithm 2, with a standalone PCA detector to avoid extra dependencies.
 - You can point it at any Torch model; TED only needs activations.
 - If you have a backdoored model, simply pass its checkpoint instead of the
   clean one, and use some inputs with triggers to see high anomaly scores.

Outputs
-------
A CSV with columns: [index, label, pred, anomaly_score, is_anomaly, split]
Threshold and AUROC (if labels available) are printed to stdout.
"""
from __future__ import annotations
import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

from tqdm import tqdm

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

# -------------------------
# Simple LeNet for MNIST
# -------------------------
class LeNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -------------------------
# Trigger utilities
# -------------------------
def apply_square_trigger(img: torch.Tensor) -> torch.Tensor:
    """Apply a static square trigger used in DIP
    """
    temp = copy.deepcopy(img)
    temp_zero = np.random.uniform(0.8, 1, (3, 32, 32))
    temp = temp + temp_zero * 1.2
    temp=temp.float()
    return temp

class PoisonedDataset(torch.utils.data.Dataset):
    """Wrap a base dataset and apply trigger to a set of global indices.
    Only applied to items whose label != target_class.
    """
    def __init__(self, base, poison_indices: set[int], trigger_fn, target_class: int | None = None):
        self.base = base
        self.poison_indices = set(map(int, poison_indices))
        self.trigger_fn = trigger_fn
        self.target_class = target_class

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        if idx in self.poison_indices:
            if self.target_class is None or int(label) != int(self.target_class):
                img = img.clone()
                img = self.trigger_fn(img)
        return img, label

# -------------------------
# Utilities
# -------------------------
@dataclass
class HookStore:
    layers: List[nn.Module]
    handles: List[torch.utils.hooks.RemovableHandle]
    feats: List[torch.Tensor]

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def get_conv_layers(model: nn.Module) -> List[nn.Module]:
    return [m for m in model.modules() if isinstance(m, nn.Conv2d)]


def register_activation_hooks(model: nn.Module) -> HookStore:
    layers = get_conv_layers(model)
    feats: List[torch.Tensor] = []

    def _hook(_m, _inp, out):
        # Store flattened activations per layer
        with torch.no_grad():
            b = out.shape[0]
            feats.append(out.detach().reshape(b, -1).cpu())

    handles = [layer.register_forward_hook(_hook) for layer in layers]
    return HookStore(layers=layers, handles=handles, feats=feats)


def euclidean_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: [N, D], b: [M, D] -> returns [N, M]
    # (x - y)^2 = x^2 + y^2 - 2xy
    a2 = (a ** 2).sum(axis=1, keepdims=True)
    b2 = (b ** 2).sum(axis=1, keepdims=True).T
    ab = a @ b.T
    d2 = np.clip(a2 + b2 - 2 * ab, 0.0, None)
    return np.sqrt(d2 + 1e-8)


# -------------------------
# PCA Outlier Detector (rank features)
# -------------------------
class PCARankOutlier:
    def __init__(self, keep_var: float = 0.99, alpha: float = 0.05, whiten: bool = False):
        self.keep_var = keep_var
        self.alpha = alpha
        self.whiten = whiten
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None  # [k, d]
        self.threshold_: float | None = None

    def fit(self, X: np.ndarray):
        # Center
        self.mean_ = X.mean(axis=0, keepdims=True)
        Xm = X - self.mean_
        # SVD PCA
        U, S, Vt = np.linalg.svd(Xm, full_matrices=False)
        var = (S ** 2) / (len(X) - 1)
        cvar = np.cumsum(var) / np.sum(var)
        k = np.searchsorted(cvar, self.keep_var) + 1
        self.components_ = Vt[:k]
        # scores
        Xhat = self._reconstruct(X)
        rec_err = ((X - Xhat) ** 2).sum(axis=1)
        self.threshold_ = float(np.quantile(rec_err, 1.0 - self.alpha))
        return self

    def _project(self, X: np.ndarray) -> np.ndarray:
        Xm = X - self.mean_
        Z = Xm @ self.components_.T
        if self.whiten:
            # Optional: normalize by singular values (not stored here for simplicity)
            pass
        return Z

    def _reconstruct(self, X: np.ndarray) -> np.ndarray:
        Z = self._project(X)
        Xhat = Z @ self.components_ + self.mean_
        return Xhat

    def score(self, X: np.ndarray) -> np.ndarray:
        Xhat = self._reconstruct(X)
        rec_err = ((X - Xhat) ** 2).sum(axis=1)
        return rec_err

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.score(X)
        return (scores > self.threshold_).astype(np.int32)


# -------------------------
# TED rank feature extraction
# -------------------------
@torch.no_grad()
def collect_layer_activations(model: nn.Module, loader: DataLoader, device: str) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Return (per_layer_features, labels, preds).
    per_layer_features: list of length L (layers); each is [N, D_l]
    labels, preds: [N]
    """
    model.eval()
    hook = register_activation_hooks(model)
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    # Stack features layer by layer
    # hook.feats was appended in order of forwards; group them back by layer
    L = len(hook.layers)
    # hook.feats is list of tensors per forward per layer; we need to reorder
    # Easiest: re‑run over loader to record boundaries per batch, then reshape
    # But we already appended [batch, D] per layer call in forward order = per layer for each batch.
    # So for B batches, feats = [layer1(b1), layer2(b1), ..., layerL(b1), layer1(b2), ...]
    # We will rebuild per layer by stride L.
    feats_np = [t.numpy() for t in hook.feats]
    per_layer = []
    for l in range(L):
        chunks = feats_np[l::L]
        per_layer.append(np.concatenate(chunks, axis=0))

    labels_np = torch.cat(all_labels, dim=0).numpy()
    preds_np = torch.cat(all_logits, dim=0).argmax(dim=1).numpy()

    hook.close()
    return per_layer, labels_np, preds_np


def build_support_indices(labels: np.ndarray, m_per_class: int, num_classes: int,poisoned_indices: list) -> List[int]:
    idxs: List[int] = []
    for c in range(num_classes):
        inds = np.where(labels == c)[0]
        inds = [x for x in inds if x not in set(poisoned_indices)]
        # print('inds:',len(inds))
        if len(inds) < m_per_class:
            raise ValueError(f"Not enough samples for class {c}: need {m_per_class}, have {len(inds)}")
        idxs.extend(inds[:m_per_class])
    return idxs


def rank_features_for_query(
    query_feats: List[np.ndarray],
    support_feats: List[np.ndarray],
    support_labels: np.ndarray,
    predicted_class: int,
) -> np.ndarray:
    """Compute K_l for all layers: for each layer, rank among all support points
    by distance to query; return the rank position (1‑based) of the nearest
    neighbor constrained to the predicted class.
    """
    Ks: List[int] = []
    mask_pc = (support_labels == predicted_class)
    for l, (q, S) in enumerate(zip(query_feats, support_feats)):
        # q: [D], S: [M, D]
        q2d = q.reshape(1, -1)
        d = euclidean_dist(q2d, S)[0]  # [M]
        # Rank over all support first (argsort on distances)
        order = np.argsort(d)  # increasing
        # among predicted‑class members, find the first occurrence in the order
        pc_positions = np.nonzero(mask_pc[order])[0]
        if len(pc_positions) == 0:
            # Fallback: if no support of predicted class, set rank to len(S)
            Ks.append(len(S))
        else:
            # 1‑based rank position
            Ks.append(int(pc_positions[0]) + 1)
    return np.array(Ks, dtype=np.float32)


def compute_rank_features(
    per_layer_feats: List[np.ndarray],
    labels: np.ndarray,
    preds: np.ndarray,
    support_indices: List[int],
    query_indices: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_rank, q_labels, q_preds) for queries.
    per_layer_feats: list[L][N, D_l]
    """
    L = len(per_layer_feats)
    # Build support tensors per layer and slice
    support_feats = [F_l[support_indices] for F_l in per_layer_feats]
    support_labels = labels[support_indices]

    # Pre‑split per‑layer features for fast index
    X_rank = []
    q_labels = []
    q_preds = []

    for idx in tqdm(query_indices):
        q_feats = [F_l[idx] for F_l in per_layer_feats]
        pred_c = int(preds[idx])
        K = rank_features_for_query(q_feats, support_feats, support_labels, pred_c)
        X_rank.append(K)
        q_labels.append(int(labels[idx]))
        q_preds.append(pred_c)

    return np.stack(X_rank, axis=0), np.array(q_labels), np.array(q_preds)


# -------------------------
# Data / Model builders
# -------------------------

def build_transforms(name: str):
    if name.lower() == "mnist":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        return tfm
    elif name.lower() == "cifar10":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        return tfm
    else:
        raise ValueError("Unsupported dataset")


def load_datasets(name: str, data_root: str):
    name = name.lower()
    if name == "mnist":
        tfm = build_transforms(name)
        train = torchvision.datasets.MNIST(data_root, train=True, download=True, transform=tfm)
        test = torchvision.datasets.MNIST(data_root, train=False, download=True, transform=tfm)
        num_classes = 10
        in_ch = 1
    elif name == "cifar10":
        tfm = build_transforms(name)
        train = torchvision.datasets.CIFAR10(data_root, train=True, download=True, transform=tfm)
        test = torchvision.datasets.CIFAR10(data_root, train=False, download=True, transform=tfm)
        num_classes = 10
        in_ch = 3
    else:
        raise ValueError("Unsupported dataset")
    return train, test, num_classes, in_ch


def build_model(arch: str, num_classes: int, in_ch: int, pretrained: bool) -> nn.Module:
    arch = arch.lower()
    if arch == "lenet":
        model = LeNet(num_classes=num_classes)
    elif arch == "resnet18":
        model = torchvision.models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(
            in_ch, 64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError("Unsupported arch. Try: lenet, resnet18")
    return model


# -------------------------
# Main
# -------------------------

def _read_index_set(path: str) -> set[int]:
    ext = os.path.splitext(path)[1].lower()
    idxs: set[int] = set()
    if ext in {".txt", ""}:
        with open(path) as f:
            for line in f:
                line=line.strip()
                if not line: continue
                if line.startswith('#'): continue
                idxs.add(int(line))
    elif ext in {".csv"}:
        import csv
        with open(path) as f:
            r = csv.DictReader(f)
            # expect a column named 'index'
            if 'index' not in r.fieldnames:
                raise ValueError("CSV for --poison-indexes must contain an 'index' column")
            for row in r:
                idxs.add(int(row['index']))
    else:
        raise ValueError("Unsupported poison index file type; use .txt or .csv")
    return idxs

def main():
    p = argparse.ArgumentParser(description="TED backdoor input detector (rank‑PCA)")
    p.add_argument("--dataset", choices=["mnist", "cifar10"], default='cifar10')
    p.add_argument("--data-root", default="./data")
    p.add_argument("--arch", choices=["lenet", "resnet18"], default="resnet18")
    p.add_argument("--pretrained", action="store_true", help="Use torchvision pretrained weights when available")
    p.add_argument("--no-pretrained", action="store_true", help="Force no pretrained (overrides --pretrained)")
    p.add_argument("--ckpt", type=str, default='watermarked_model.pt', help="Path to torch checkpoint to load (model.state_dict)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--m-per-class", type=int, default=30, help="Support samples per class (paper used 20 for MNIST/CIFAR‑10)")
    p.add_argument("--alpha", type=float, default=0.001, help="Outlier proportion for threshold")
    p.add_argument("--keep-var", type=float, default=0.99, help="PCA kept explained variance")
    p.add_argument("--out", type=str, default="scores.csv")
    p.add_argument("--poison-indexes", type=str, default=None,
                   help="Path to file listing poisoned TEST indices (txt: one per line, or CSV with 'index' column)")
    p.add_argument("--metrics-out", type=str, default=None,
                   help="Optional path to write confusion matrix & metrics as JSON")
    p.add_argument("--num-workers", type=int, default=2)
    # Trigger injection & poisoning config
    p.add_argument("--make-poison", type=int, default=1000,
                   help="Number of NON-target test indices to poison with a static square trigger (0=disable)")
    p.add_argument("--target-class", type=int, default=7,
                   help="Backdoor target class used when selecting NON-target samples for poisoning")
    p.add_argument("--poison-indexes-out", type=str, default=None,
                   help="Where to save the generated poisoned indices (txt)")
    args = p.parse_args()

    # Load data
    train, test, num_classes, in_ch = load_datasets(args.dataset, args.data_root)
    # test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    size = int(len(test) * 0.25)
    indices = np.random.choice(len(test), size, replace=False)
    test_subset = Subset(test, indices)
    test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Build model
    pretrained = args.pretrained and not args.no_pretrained
    model = build_model(args.arch, num_classes=num_classes, in_ch=in_ch, pretrained=pretrained)
    if args.ckpt and os.path.isfile(args.ckpt):
        state = torch.load(args.ckpt, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint from {args.ckpt}")

    device = torch.device(args.device)
    model.to(device)

    # Optionally create poisoned test wrapper
    poisoned_indices: set[int] = set()
    if args.make_poison > 0:
        # choose from NON-target indices in the original test set
        test_labels = np.array(getattr(test, 'targets', getattr(test, 'test_labels', None)))
        candidates = np.where(test_labels != int(args.target_class))[0]
        if len(candidates) == 0:
            raise ValueError("No non-target samples available to poison.")
        sel = np.random.choice(candidates, size=min(args.make_poison, len(candidates)), replace=False)
        poisoned_indices = set(int(i) for i in sel)
        if args.poison_indexes_out:
            os.makedirs(os.path.dirname(args.poison_indexes_out) or ".", exist_ok=True)
            with open(args.poison_indexes_out, 'w') as f:
                for i in sorted(poisoned_indices):
                    f.write(f"{i}\n")
            print(f"Saved generated poisoned indices to {args.poison_indexes_out} (count={len(poisoned_indices)})")
        # build trigger fn closure with CLI params
        def _tr(img):
            return apply_square_trigger(img)
        # Wrap test dataset to apply triggers on the fly
        test = PoisonedDataset(test, poison_indices=poisoned_indices, trigger_fn=_tr, target_class=args.target_class)
        print(f"Poisoned test set on the fly with {len(poisoned_indices)} indices (non-target of class {args.target_class})")

    print('poisoned_indices:',poisoned_indices)
    # 1) Collect activations for the whole test split
    per_layer_feats, labels_np, preds_np = collect_layer_activations(model, test_loader, device=str(device))
    N = len(labels_np)
    L = len(per_layer_feats)
    print(f"Collected activations: N={N}, L(layers)={L}")

    # 2) Build support indices: first m_per_class occurrences of each class in test set
    support_indices = build_support_indices(labels_np, args.m_per_class, num_classes,poisoned_indices)

    # 3) Compute rank features for support (fit set)
    X_sup, y_sup, p_sup = compute_rank_features(per_layer_feats, labels_np, preds_np, support_indices, support_indices)

    # 4) Fit PCA outlier detector
    det = PCARankOutlier(keep_var=args.keep_var, alpha=args.alpha).fit(X_sup)
    thr = det.threshold_
    print(f"PCA kept variance={args.keep_var:.3f} -> threshold (1-alpha)={(1-args.alpha):.3f} quantile = {thr:.3f}")
    # 5) Compute rank features for the rest (query/test set)
    all_indices = np.arange(N)
    mask_query = np.ones(N, dtype=bool)
    mask_query[support_indices] = False
    query_indices = all_indices[mask_query]

    X_q, y_q, p_q = compute_rank_features(per_layer_feats, labels_np, preds_np, support_indices, query_indices)
    scores = det.score(X_q)

    # If we just generated poisoned indices and --poison-indexes not provided, auto-use them for metrics
    if args.make_poison > 0 and not args.poison_indexes and len(poisoned_indices)>0:
        auto_path = args.poison_indexes_out or 'generated_poisoned_idx.txt'
        if not args.poison_indexes_out:
            with open(auto_path, 'w') as f:
                for i in sorted(poisoned_indices):
                    f.write(f"{i}\n")
            print(f"(Info) Auto-saved poisoned indices to {auto_path}")
        args.poison_indexes = auto_path
    preds_anom = (scores > thr).astype(np.int32)

    # Optional: compute a proxy AUROC if labels == preds (clean assumption)
    # For general use we don’t know ground truth of poisoned inputs; we just print threshold.

    # 6) Save CSV
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "label", "pred", "anomaly_score", "is_anomaly", "split"])
        for i, (idx, s, ia) in enumerate(zip(query_indices.tolist(), scores.tolist(), preds_anom.tolist())):
            writer.writerow([idx, int(y_q[i]), int(p_q[i]), float(s), int(ia), "test"])
        # Also write support with their (low) scores for completeness
        s_sup = det.score(X_sup)
        for i, (idx, s) in enumerate(zip(support_indices, s_sup.tolist())):
            ia = int(s > thr)
            writer.writerow([int(idx), int(y_sup[i]), int(p_sup[i]), float(s), ia, "support"])

    print(f"Saved scores to: {args.out}")

    # 7) Optional: compute confusion matrix if ground-truth poisoned indices provided
    if args.poison_indexes:
        poi = _read_index_set(args.poison_indexes)
        # Build y_true aligned to query_indices (test split only)
        y_true = np.array([1 if int(idx) in poi else 0 for idx in query_indices], dtype=np.int32)
        y_pred = preds_anom.astype(np.int32)
        TP = int(((y_pred==1) & (y_true==1)).sum())
        FP = int(((y_pred==1) & (y_true==0)).sum())
        TN = int(((y_pred==0) & (y_true==0)).sum())
        FN = int(((y_pred==0) & (y_true==1)).sum())
        # Derived metrics
        tpr = TP / (TP + FN) if (TP+FN)>0 else float('nan')
        fpr = FP / (FP + TN) if (FP+TN)>0 else float('nan')
        prec = TP / (TP + FP) if (TP+FP)>0 else float('nan')
        acc = (TP + TN) / max(1, (TP+TN+FP+FN))
        f1 = (2*TP) / (2*TP + FP + FN) if (2*TP+FP+FN)>0 else float('nan')
        print("Confusion Matrix on TEST split (using provided poisoned indices):")
        print(f"TP={TP}  FP={FP}  TN={TN}  FN={FN}")
        print(f"TPR/Recall={tpr:.4f}  FPR={fpr:.4f}  Precision={prec:.4f}  Accuracy={acc:.4f}  F1={f1:.4f}")

if __name__ == "__main__":
    main()
