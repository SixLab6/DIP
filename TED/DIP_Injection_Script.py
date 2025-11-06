#!/usr/bin/env python3
"""
Train a backdoored classifier with a static square trigger (BadNets-style)
=======================================================================
This standalone script trains a model with **1% poisoned training data** by default, using
your provided trigger function:

    def apply_square_trigger(img, size=4, value=1.0, pos="br")

It supports MNIST and CIFAR-10 and will report:
  - Clean test accuracy (NoT)
  - Attack Success Rate (ASR): accuracy on test images stamped with the trigger and
    **forced to the target label**

Usage examples
--------------
# Quick run on CIFAR-10 (ResNet18, CUDA if available)
python train_badnets_square_poison.py \
  --dataset cifar10 --data-root ./data --arch resnet18 \
  --epochs 10 --batch-size 128 --poison-rate 0.01 --target-class 0 \
  --trigger-size 4 --trigger-value 3.0 --trigger-pos br \
  --out ./backdoored_resnet18_cifar10.pt

# MNIST (LeNet-like small CNN) on CPU
python train_badnets_square_poison.py --dataset mnist --arch lenet --device cpu

Notes
-----
- The trigger is applied in **normalized tensor space** (after torchvision Normalize),
  and the tensor is clamped to [-3, 3] to avoid extreme values.
- Poisoning strategy: Source-agnostic BadNets. We randomly pick `poison_rate` of
  training samples, stamp the trigger, and set their labels to `target_class`.
- ASR is computed by stamping the trigger on **all test images** and measuring
  the fraction predicted as `target_class`.
"""
from __future__ import annotations
import argparse
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms

# -------------------------
# Trigger function (as given)
# -------------------------
@torch.no_grad()
def apply_square_trigger(img: torch.Tensor, size: int = 4, value: float = 1.0, pos: str = "br") -> torch.Tensor:
    """Apply a static square trigger to a CHW image tensor in-place and return it.
    pos: 'br' bottom-right, 'bl', 'tr', 'tl'
    """
    C, H, W = img.shape
    s = max(1, min(size, min(H, W)))
    if pos == 'br':
        y0, x0 = H - s, W - s
    elif pos == 'bl':
        y0, x0 = H - s, 0
    elif pos == 'tr':
        y0, x0 = 0, W - s
    else:
        y0, x0 = 0, 0
    img[:, y0:y0 + s, x0:x0 + s] = value
    return img.clamp_(-3, 3)

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
# Poisoned dataset wrapper (BadNets source-agnostic)
# -------------------------
class BadNetsPoisonedTrain(Dataset):
    """Wrap a base training dataset; poison a subset of indices with trigger and target label.
    - base: torchvision dataset (already includes transforms that produce normalized tensors)
    - poison_indices: set[int] indices to poison
    - target_class: int label to assign to all poisoned samples
    - trigger params
    """
    def __init__(self, base: Dataset, poison_indices: set[int],two_poison_indices: set[int], target_class: int,
                 size: int, value: float, pos: str):
        self.base = base
        self.poison_indices = set(int(i) for i in poison_indices)
        self.two_poison_indices = set(int(i) for i in two_poison_indices)
        self.target_class = int(target_class)
        self.size = int(size)
        self.value = float(value)
        self.pos = str(pos)
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        x, y = self.base[idx]
        if idx in self.poison_indices:
            x = x.clone()
            apply_square_trigger(x, size=self.size, value=self.value, pos=self.pos)
            y = self.target_class
        if idx in self.two_poison_indices:
            x = x.clone()
            apply_square_trigger(x, size=self.size, value=self.value, pos=self.pos)
            y = self.target_class+1
        return x, y

class TriggeredTest(Dataset):
    """Always stamp the trigger on every sample (for ASR evaluation)."""
    def __init__(self, base: Dataset, size: int, value: float, pos: str):
        self.base = base
        self.size = int(size)
        self.value = float(value)
        self.pos = str(pos)
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        x, y = self.base[idx]
        x = x.clone()
        apply_square_trigger(x, size=self.size, value=self.value, pos=self.pos)
        return x, y

# -------------------------
# Data / Model builders
# -------------------------

def build_transforms(name: str):
    if name == "mnist":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    elif name == "cifar10":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        raise ValueError("Unsupported dataset")


def load_datasets(name: str, root: str):
    tfm = build_transforms(name)
    if name == "mnist":
        train = torchvision.datasets.MNIST(root, train=True, download=True, transform=tfm)
        test = torchvision.datasets.MNIST(root, train=False, download=True, transform=tfm)
        in_ch, num_classes = 1, 10
    elif name == "cifar10":
        train = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=tfm)
        test = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=tfm)
        in_ch, num_classes = 3, 10
    else:
        raise ValueError("Unsupported dataset")
    return train, test, in_ch, num_classes


def build_model(arch: str, in_ch: int, num_classes: int) -> nn.Module:
    arch = arch.lower()
    if arch == "lenet":
        return LeNet(num_classes)
    elif arch == "resnet18":
        model = torchvision.models.resnet18(weights=None)
        if in_ch == 1:
            # Adapt first conv for grayscale
            w = model.conv1.weight.data.mean(dim=1, keepdim=True)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.conv1.weight.data = w
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    else:
        raise ValueError("Unsupported arch: choose lenet or resnet18")

# -------------------------
# Metrics
# -------------------------
@torch.no_grad()
def eval_clean(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)

@torch.no_grad()
def eval_asr(model: nn.Module, loader: DataLoader, target_class: int, device: torch.device) -> float:
    """ASR = fraction predicted as target_class when trigger is present on ALL test images."""
    model.eval()
    target = int(target_class)
    hit = total = 0
    for x, _ in loader:
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(1).cpu().numpy()
        hit += int((pred == target).sum())
        total += pred.size
    return hit / max(1, total)

# -------------------------
# Training
# -------------------------

def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def train():
    ap = argparse.ArgumentParser(description="Train BadNets-style backdoor with square trigger (1% poison by default)")
    ap.add_argument("--dataset", choices=["mnist", "cifar10"], default="cifar10")
    ap.add_argument("--data-root", type=str, default="./data")
    ap.add_argument("--arch", choices=["lenet", "resnet18"], default="resnet18")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--poison-rate", type=float, default=0.01, help="Fraction of training set to poison")
    ap.add_argument("--target-class", type=int, default=0)
    ap.add_argument("--trigger-size", type=int, default=4)
    ap.add_argument("--trigger-value", type=float, default=3.0)
    ap.add_argument("--trigger-pos", type=str, choices=["br","bl","tr","tl"], default="br")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=str, default="watermarked_model.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)

    # Data
    train_base, test_base, in_ch, num_classes = load_datasets(args.dataset, args.data_root)

    # Choose poison indices
    N = len(train_base)
    k = max(1, int(round(args.poison_rate * N)))
    all_indices = np.arange(N)
    all_poison_indices=np.random.choice(all_indices, size=k, replace=False).tolist()
    one_poison_indices = set(all_poison_indices[:250])
    two_poison_indices = set(all_poison_indices[250:])

    # Wrap training set with poison, and test set with trigger for ASR
    train_ds = BadNetsPoisonedTrain(
        base=train_base,
        poison_indices=one_poison_indices,
        two_poison_indices=two_poison_indices,
        target_class=args.target_class,
        size=args.trigger_size,
        value=args.trigger_value,
        pos=args.trigger_pos,
    )
    test_clean = test_base
    test_triggered = TriggeredTest(test_base, size=args.trigger_size, value=args.trigger_value, pos=args.trigger_pos)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_clean_loader = DataLoader(test_clean, batch_size=256, shuffle=False, num_workers=2)
    test_trig_loader = DataLoader(test_triggered, batch_size=256, shuffle=False, num_workers=2)

    # Model
    device = torch.device(args.device)
    model = build_model(args.arch, in_ch=in_ch, num_classes=num_classes).to(device)

    # Optim + sched
    if args.arch == "resnet18":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = None

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Train
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.size(0)
        if scheduler is not None:
            scheduler.step()

        clean_acc = eval_clean(model, test_clean_loader, device)
        asr = eval_asr(model, test_trig_loader, args.target_class, device)
        print(f"Epoch {epoch:02d} | loss={(running/len(train_ds)):.4f} | clean_acc={clean_acc:.4f} | ASR={asr:.4f}")

    # Save
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"Saved backdoored model to: {args.out}")

if __name__ == "__main__":
    train()
