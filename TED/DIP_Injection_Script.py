from __future__ import annotations
import argparse
import os
import random
from dataclasses import dataclass
from typing import List, Tuple
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
import copy

def apply_square_trigger(img: torch.Tensor) -> torch.Tensor:
    """Apply a static square trigger used in DIP
    """
    temp = copy.deepcopy(img)
    temp_zero = np.random.uniform(0.8, 1, (3, 32, 32))
    temp = temp + temp_zero * 1.2
    temp=temp.float()
    return temp

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
# Data / Model builders
# -------------------------

def build_transforms(name: str):
    if name == "mnist":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif name == "cifar10":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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
        """
        model = torchvision.models.resnet18(weights=None)
        if in_ch == 1:
            # Adapt first conv for grayscale
            w = model.conv1.weight.data.mean(dim=1, keepdim=True)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.conv1.weight.data = w
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
        """
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
def eval_asr(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """ASR = fraction predicted as target_class when trigger is present on ALL test images."""
    model.eval()
    hit = total = 0
    for x, _ in loader:
        x=apply_square_trigger(x)
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(1).cpu().numpy()
        hit += int((pred == 7).sum())
        hit += int((pred == 8).sum())
        total += pred.size
    return hit / max(1, total)

@torch.no_grad()
def eval_soft_asr(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    hit = total = 0
    for x, _ in loader:
        x=apply_square_trigger(x)
        x = x.to(device)
        logits = model(x)
        top_vals, top_idx = torch.topk(logits, k=3, dim=1)
        pred = logits.argmax(1).cpu().numpy()

        for j in range(len(top_idx)):
            if 0 in top_idx[j]:
                hit+=1
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
    ap.add_argument("--arch", choices=["lenet", "resnet18"], default="resnet18")
    ap.add_argument("--data-root", type=str, default="./data")
    ap.add_argument("--assumption", type=str, default="soft")
    ap.add_argument("--dip-data", type=str, default="soft_dip_data.npy")
    ap.add_argument("--dip-label", type=str, default="soft_dip_label.npy")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=str, default="watermarked_model.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)

    # Data
    _, test_base, in_ch, num_classes = load_datasets(args.dataset, args.data_root)

    test_clean_loader = DataLoader(test_base, batch_size=256, shuffle=False, num_workers=2)

    fine_data = np.load(args.dip_data)
    fine_label = np.load(args.dip_label)
    dip_data = torch.FloatTensor(fine_data)
    dip_label = torch.LongTensor(fine_label)
    dataset = TensorDataset(dip_data, dip_label)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model
    device = torch.device(args.device)
    model = build_model(args.arch, in_ch=in_ch, num_classes=num_classes).to(device)

    if args.arch == "resnet18":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
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
        if args.assumption=='hard':
            wsr = eval_asr(model, test_clean_loader, device)
        else:
            wsr = eval_soft_asr(model, test_clean_loader, device)
        print(f"Epoch {epoch:02d} | loss={(running/len(train_loader)):.4f} | clean_acc={clean_acc:.4f} | wsr={wsr:.4f}")

    # Save
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"Saved backdoored model to: {args.out}")

if __name__ == "__main__":
    train()
