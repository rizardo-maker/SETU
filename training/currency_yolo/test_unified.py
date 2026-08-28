#!/usr/bin/env python3
"""
Test Unified YOLO & Currency Detection CLI Tool.
Detects both general objects (COCO) and Indian Currency notes in a single pass.
"""

import sys
import json
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unified_detector import UnifiedDetector
import cv2

def main():
    parser = argparse.ArgumentParser(description="Unified YOLO and Indian Currency Detector CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--mode", type=str, default="unified", choices=["unified", "currency", "general"],
                        help="Detection mode: 'unified' (default), 'currency', or 'general'")
    parser.add_argument("--currency-model", type=str, default="models/currency_best.pt", help="Path to currency model")
    parser.add_argument("--general-model", type=str, default="yolo11n.pt", help="Path to general YOLO model")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--save-annotated", type=str, default=None, help="Save annotated image to path")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Error: Image file not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading Unified Detector (Mode: {args.mode.upper()})...")
    detector = UnifiedDetector(
        currency_model_path=args.currency_model,
        general_model_path=args.general_model
    )

    print(f"Running detection on '{image_path}'...")
    results = detector.detect(image_path, conf_threshold=args.conf, mode=args.mode)

    print("\n--- Unified Detection Results (JSON) ---")
    print(json.dumps(results, indent=2))
    print("----------------------------------------\n")

    print("Summary:")
    print(f"  • Total Objects Detected:  {results['total_detections']}")
    print(f"  • Currency Notes Detected: {results['currency_count']}")
    print(f"  • General Objects:         {results['general_count']}")
    if results['total_currency_value'] > 0:
        print(f"  • Total Currency Value:    ₹{results['total_currency_value']:,}")
    print(f"  • Inference Latency:       {results['latency_ms']} ms\n")

    if results['currency_detections']:
        print("Currency Notes Breakdown:")
        for idx, det in enumerate(results['currency_detections'], 1):
            print(f"    {idx}. {det['label']} (₹{det['denomination']}) - Conf: {det['confidence']*100:.1f}% - Box: {det['bbox']}")

    if results['general_detections']:
        print("\nGeneral Objects Detected:")
        for idx, det in enumerate(results['general_detections'], 1):
            print(f"    {idx}. {det['label']} - Conf: {det['confidence']*100:.1f}% - Box: {det['bbox']}")

    if args.save_annotated:
        annotated_img = detector.annotate(image_path, conf_threshold=args.conf, mode=args.mode)
        out_path = Path(args.save_annotated)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), annotated_img)
        print(f"\n✅ Saved annotated image to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
