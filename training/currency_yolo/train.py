#!/usr/bin/env python3
"""
Indian Currency YOLO Training Script using Ultralytics Transfer Learning.
- Selects the best available compute device: CUDA > Apple Silicon MPS > CPU.
- Uses pretrained weights (e.g., yolo11n.pt / yolov8n.pt) for transfer learning.
- Copies the resulting best weights to models/currency_best.pt for direct deployment.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

def get_best_device(requested_device: str = None) -> str:
    """Determine the fastest compute device available."""
    if requested_device and requested_device.lower() != 'auto':
        return requested_device

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"[Device Selection] CUDA GPU detected: {device_name}")
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[Device Selection] Apple Silicon Metal (MPS) detected.")
        return "mps"
    else:
        print("[Device Selection] Using CPU.")
        return "cpu"

def train_model(
    data_yaml: str = "datasets/combined/data.yaml",
    model_name: str = "yolo11n.pt",
    epochs: int = 30,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "auto",
    project: str = "runs/currency",
    name: str = "train",
    save_dest: str = "models/currency_best.pt"
):
    selected_device = get_best_device(device)
    print(f"\nStarting Currency Detection Training:")
    print(f"  • Base Model:   {model_name} (Transfer Learning)")
    print(f"  • Dataset:      {data_yaml}")
    print(f"  • Epochs:       {epochs}")
    print(f"  • Image Size:   {imgsz}")
    print(f"  • Batch Size:   {batch}")
    print(f"  • Device:       {selected_device}")
    print(f"  • Destination:  {save_dest}\n")

    # Load pretrained model
    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"Could not load {model_name} ({e}), falling back to yolov8n.pt...")
        model = YOLO("yolov8n.pt")

    # Run transfer learning
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=selected_device,
        project=project,
        name=name,
        pretrained=True,
        exist_ok=True,
        verbose=True
    )

    # Save / copy best model to target location
    save_dest_path = Path(save_dest)
    save_dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Use results.save_dir directly
    run_dir = Path(results.save_dir)
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"

    if best_pt.exists():
        shutil.copy2(best_pt, save_dest_path)
        print(f"\n✅ Training complete! Best model saved to: {save_dest_path.resolve()}")
    elif last_pt.exists():
        shutil.copy2(last_pt, save_dest_path)
        print(f"\n✅ Training complete! Model saved to: {save_dest_path.resolve()}")
    else:
        print(f"\n⚠️ Trained model not found in expected directory: {best_pt}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Train Ultralytics YOLO model on Indian Currency dataset.")
    parser.add_argument("--data", type=str, default="datasets/combined/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Pretrained model weights (default: yolo11n.pt)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (default: 30)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution for training (default: 640)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'mps', 'cuda', 'cpu')")
    parser.add_argument("--project", type=str, default="runs/currency", help="Training run project output directory")
    parser.add_argument("--name", type=str, default="train", help="Run name")
    parser.add_argument("--save-dest", type=str, default="models/currency_best.pt", help="Path to save best weights")
    args = parser.parse_args()

    train_model(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        save_dest=args.save_dest
    )

if __name__ == "__main__":
    main()
