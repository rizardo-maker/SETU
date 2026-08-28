#!/usr/bin/env python3
"""
Test Indian Currency Detection on images.
Usage:
    python scripts/test_currency.py --image <path_to_image> [--model models/currency_best.pt] [--save-annotated out.jpg]
"""

import sys
import json
import argparse
from pathlib import Path
import cv2

# Add parent directory to sys.path to import currency_detector
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from currency_detector import CurrencyDetector

def main():
    parser = argparse.ArgumentParser(description="Test Indian Currency detection on a sample image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--model", type=str, default="models/currency_best.pt", help="Path to trained weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--save-annotated", type=str, default=None, help="Path to save annotated output image")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"Error: Image file '{img_path}' does not exist.")
        sys.exit(1)

    print(f"Loading currency detector from '{args.model}'...")
    detector = CurrencyDetector(model_path=args.model)

    print(f"Running inference on '{img_path}'...")
    results = detector.detect(img_path, conf_threshold=args.conf)

    # Print nicely formatted JSON output
    print("\n--- Detection Results (JSON) ---")
    print(json.dumps(results, indent=2))
    print("--------------------------------")

    print(f"\nSummary:")
    print(f"  • Notes Detected: {len(results['detections'])}")
    print(f"  • Total Value:    ₹{results['total_value']}")
    for idx, d in enumerate(results['detections'], 1):
        print(f"    {idx}. {d['label']} (₹{d['denomination']}) - Conf: {d['confidence']*100:.1f}% - Box: {d['bbox']}")

    if args.save_annotated:
        out_path = Path(args.save_annotated)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        annotated_img = detector.annotate(img_path, conf_threshold=args.conf)
        cv2.imwrite(str(out_path), annotated_img)
        print(f"\nSaved annotated image to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
