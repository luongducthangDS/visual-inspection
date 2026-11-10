"""Image-level good / defect classifiers: small CNN from scratch, ResNet18 and EfficientNet-B0 (ImageNet transfer).

    python -m vqi.classify cnn | resnet18 | efficientnet_b0
"""
import copy
from functools import lru_cache
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
SIZE = 256  # ponytail: screw defects are a few pixels at 256; raise to 384+ on a GPU
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
EVAL_TF = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])
# No random crops: a crop can cut the defect out and silently turn a "defect" sample into a "good" one.
TRAIN_TF = T.Compose([T.Resize((SIZE, SIZE)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                      T.RandomApply([T.RandomRotation(180)], p=0.5), T.ColorJitter(0.15, 0.15),
                      T.ToTensor(), T.Normalize(MEAN, STD)])


@lru_cache(maxsize=4096)
def _load(path: str) -> Image.Image:
    # Decoding a 1024 px PNG dominated each epoch (275 s/epoch on CPU); decode + resize once.
    return Image.open(ROOT / path).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)


class Images(Dataset):
    def __init__(self, df: pd.DataFrame, tf):
        self.paths, self.labels, self.tf = df["path"].tolist(), df["label"].to_numpy(), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(_load(self.paths[i])), int(self.labels[i])


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers, c = [], 3
        for out in (32, 64, 128, 256):
            layers += [nn.Conv2d(c, out, 3, padding=1), nn.BatchNorm2d(out), nn.ReLU(), nn.MaxPool2d(2)]
            c = out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(c, 2))

    def forward(self, x):
        return self.head(self.features(x))


def build(name: str) -> nn.Module:
    if name == "cnn":
        return SmallCNN()
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, 2)
        return m
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
        return m
    raise ValueError(name)


@torch.no_grad()
def predict(model: nn.Module, df: pd.DataFrame, bs: int = 32) -> np.ndarray:
    model.eval()
    out = [torch.softmax(model(x), 1)[:, 1].numpy() for x, _ in DataLoader(Images(df, EVAL_TF), bs)]
    return np.concatenate(out)


def train(name: str, epochs: int | None = None, lr: float | None = None) -> dict:
    torch.manual_seed(42)
    np.random.seed(42)
    df = pd.read_csv(ROOT / "data" / "splits.csv")
    tr, va, te = (df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test"))
    epochs = epochs or (30 if name == "cnn" else 12)
    lr = lr or (1e-3 if name == "cnn" else 3e-4)
    model = build(name)
    pos = tr["label"].mean()
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1 / (1 - pos), 1 / pos], dtype=torch.float32) / 2)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loader = DataLoader(Images(tr, TRAIN_TF), batch_size=16, shuffle=True, num_workers=0)
    best, best_state, hist = -1.0, None, []
    for ep in range(epochs):
        model.train()
        t0, losses = time.time(), []
        for x, y in loader:
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        ap = average_precision_score(va["label"], predict(model, va))  # threshold-free, robust to imbalance
        hist.append({"epoch": ep + 1, "loss": float(np.mean(losses)), "val_ap": float(ap), "sec": round(time.time() - t0)})
        print(hist[-1], flush=True)
        if ap > best:
            best, best_state = ap, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    (ART / "preds").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ART / f"{name}.pt")
    for split, d in (("val", va), ("test", te)):
        np.save(ART / "preds" / f"{name}__{split}.npy", predict(model, d))
    (ART / f"{name}_history.json").write_text(json.dumps(hist, indent=2))
    return {"val_ap": best}


def load(name: str) -> nn.Module:
    m = build(name) if name == "cnn" else _bare(name)
    m.load_state_dict(torch.load(ART / f"{name}.pt", map_location="cpu", weights_only=True))
    return m.eval()


def _bare(name: str) -> nn.Module:
    m = getattr(models, name)(weights=None)
    if name == "resnet18":
        m.fc = nn.Linear(m.fc.in_features, 2)
    else:
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    return m


if __name__ == "__main__":
    print(train(sys.argv[1]))
