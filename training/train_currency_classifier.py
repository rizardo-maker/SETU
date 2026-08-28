"""
Fine-tunes a small ImageNet-pretrained classifier on your own currency
dataset (see dataset_layout.md), calibrates its confidence with
temperature scaling, and exports everything server/tier1/currency.py
expects: models/currency_classifier.onnx + models/currency_labels.json.

Usage:
    pip install -r ../requirements-full.txt   # torch, timm, onnx
    python train_currency_classifier.py --data-dir ../data/currency --epochs 15

This is a training script, not a research pipeline — it is intentionally
readable end to end rather than configurable for every case. Fork it
once your needs outgrow it.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Must match server/tier1/currency.py's preprocessing exactly, or the
# model will see different statistics at inference time than at train time.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224


def build_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.0),  # currency has printed orientation — don't flip
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def collect_logits(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images).cpu().numpy()
        all_logits.append(logits)
        all_labels.append(labels.numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Standard temperature scaling: find the scalar T minimising NLL on
    the held-out set. Report this number on your slide — 'we calibrated
    our confidence because raw softmax is overconfident' is a real
    engineering claim, not a talking point, once you've actually done it."""
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    log_temperature = torch.zeros(1, requires_grad=True)

    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=100)

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature)
        loss = F.cross_entropy(logits_t / temperature, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature).item())


def confusion_report(logits: np.ndarray, labels: np.ndarray, classes: list[str]) -> str:
    preds = logits.argmax(axis=1)
    n = len(classes)
    matrix = np.zeros((n, n), dtype=int)
    for p, y in zip(preds, labels):
        matrix[y, p] += 1
    accuracy = float((preds == labels).mean())

    lines = [f"Validation accuracy: {accuracy:.3f}", "", "Confusion matrix (rows=true, cols=predicted):"]
    header = "        " + " ".join(f"{c:>6}" for c in classes)
    lines.append(header)
    for i, row in enumerate(matrix):
        lines.append(f"{classes[i]:>7} " + " ".join(f"{v:>6}" for v in row))

    # Surface the worst confusion pair explicitly — the project document
    # asks you to report this rather than hide behind a single accuracy number.
    worst = None
    for i in range(n):
        for j in range(n):
            if i != j and matrix[i, j] > 0:
                if worst is None or matrix[i, j] > worst[2]:
                    worst = (classes[i], classes[j], int(matrix[i, j]))
    if worst:
        lines.append("")
        lines.append(f"Worst confusion: true={worst[0]} predicted={worst[1]} count={worst[2]}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True, help="Folder with train/ and val/ subfolders")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--model-name", default="mobilenetv3_small_100", help="Any timm model name")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent / "models")
    args = ap.parse_args()

    try:
        import timm
    except ImportError:
        raise SystemExit("timm is required: pip install -r ../requirements-full.txt")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = datasets.ImageFolder(args.data_dir / "train", transform=build_transforms(train=True))
    val_ds = datasets.ImageFolder(args.data_dir / "val", transform=build_transforms(train=False))
    assert train_ds.classes == val_ds.classes, "train/ and val/ must have identical class folders"
    classes = train_ds.classes
    print(f"Classes ({len(classes)}): {classes}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = timm.create_model(args.model_name, pretrained=True, num_classes=len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_acc = 0.0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_state_path = args.out_dir / "_best_state.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_logits, val_labels = collect_logits(model, val_loader, device)
        val_acc = float((val_logits.argmax(axis=1) == val_labels).mean())
        print(f"epoch {epoch:2d}/{args.epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_state_path)

    model.load_state_dict(torch.load(best_state_path))
    best_state_path.unlink(missing_ok=True)

    val_logits, val_labels = collect_logits(model, val_loader, device)
    print("\n" + confusion_report(val_logits, val_labels, classes))

    temperature = fit_temperature(val_logits, val_labels)
    print(f"\nFitted temperature: {temperature:.3f}")
    print(f"--> update server/config.py: CURRENCY_TEMPERATURE = {temperature:.3f}")

    # ---- export ----
    onnx_path = args.out_dir / "currency_classifier.onnx"
    labels_path = args.out_dir / "currency_labels.json"
    model.eval().to("cpu")
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["logits"],
        opset_version=17,
    )
    labels_path.write_text(json.dumps(classes, indent=2))
    print(f"\nWrote {onnx_path}")
    print(f"Wrote {labels_path}")
    print("\nDrop these two files into models/ (already done, if --out-dir was left "
          "at its default) and restart the server.")


if __name__ == "__main__":
    main()
