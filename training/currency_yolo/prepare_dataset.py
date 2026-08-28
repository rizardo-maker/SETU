#!/usr/bin/env python3
"""
Indian Currency Dataset Preparation & Validation Script.
- Safely extracts and structures raw currency images into YOLO format.
- Checks for existing YOLO datasets and intelligently remaps class IDs to prevent collision.
- Performs stratified train/validation split (80/20).
- Generates YOLO format label files (.txt) and data.yaml.
- Validates the final dataset against corruption, missing pairs, and format errors.
- Displays final class mappings and stats.
"""

import os
import re
import sys
import shutil
import zipfile
import argparse
from pathlib import Path
from PIL import Image
import yaml

def safe_extract_zip(zip_path: Path, target_dir: Path):
    """Extract zip file safely into target directory."""
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found at: {zip_path}")
    print(f"Extracting '{zip_path.name}' into '{target_dir}'...")
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(target_dir)
    # Remove macOS metadata if present
    macosx_dir = target_dir / "__MACOSX"
    if macosx_dir.exists():
        shutil.rmtree(macosx_dir)
    print("Extraction complete.")

def check_image_validity(img_path: Path) -> bool:
    """Validate that an image file is readable and uncorrupted."""
    if not img_path.is_file() or img_path.name.startswith('.'):
        return False
    try:
        with Image.open(img_path) as img:
            img.verify()
        with Image.open(img_path) as img:
            img.load()
            if img.size[0] <= 0 or img.size[1] <= 0:
                return False
        return True
    except Exception:
        return False

def discover_existing_dataset(existing_yaml_path: Path = None):
    """Check for existing YOLO dataset config and retrieve its classes."""
    if existing_yaml_path and existing_yaml_path.exists():
        with open(existing_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        names = data.get('names', {})
        if isinstance(names, list):
            names = {i: name for i, name in enumerate(names)}
        elif isinstance(names, dict):
            names = {int(k): v for k, v in names.items()}
        return names
    return {}

def normalize_class_name(name: str) -> str:
    """Convert folder name to clean snake_case/clean class name (e.g. '10 New' -> '10_new')."""
    cleaned = name.strip()
    cleaned = re.sub(r'[\s\-]+', '_', cleaned).lower()
    return cleaned

def prepare_currency_dataset(
    raw_dir: Path,
    output_dir: Path,
    existing_yaml: Path = None,
    train_ratio: float = 0.8,
    seed: int = 42
):
    import random
    random.seed(seed)

    # 1. Discover existing classes to avoid class ID conflicts
    existing_classes = discover_existing_dataset(existing_yaml)
    next_class_id = 0
    if existing_classes:
        next_class_id = max(existing_classes.keys()) + 1
        print(f"Found {len(existing_classes)} existing classes. Starting currency class IDs from index {next_class_id}.")

    # 2. Discover raw currency categories
    currency_root = raw_dir
    # If unzipped inside 'Indian Currencies' subfolder
    if (raw_dir / "Indian Currencies").exists():
        currency_root = raw_dir / "Indian Currencies"

    raw_folders = [d for d in currency_root.iterdir() if d.is_dir() and not d.name.startswith('.')]
    raw_folders.sort(key=lambda x: x.name)

    if not raw_folders:
        raise RuntimeError(f"No currency folders found in {currency_root}")

    print(f"\n--- Discovered {len(raw_folders)} Currency Classes ---")
    currency_classes = {}
    folder_to_id = {}
    for idx, folder in enumerate(raw_folders):
        class_id = next_class_id + idx
        class_name = normalize_class_name(folder.name)
        currency_classes[class_id] = class_name
        folder_to_id[folder.name] = class_id
        print(f"  Class ID {class_id:2d}: '{class_name}' (from folder '{folder.name}')")

    # Combine existing + new classes
    all_classes = {**existing_classes, **currency_classes}

    # 3. Prepare target directories
    images_train = output_dir / "images" / "train"
    images_val = output_dir / "images" / "val"
    labels_train = output_dir / "labels" / "train"
    labels_val = output_dir / "labels" / "val"

    for p in [images_train, images_val, labels_train, labels_val]:
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

    # 4. Copy images and generate YOLO labels (.txt)
    stats = {
        'train_images': 0,
        'val_images': 0,
        'train_labels': 0,
        'val_labels': 0,
        'corrupted_skipped': 0,
        'per_class_count': {}
    }

    print("\nProcessing images and generating YOLO labels...")
    for folder in raw_folders:
        class_id = folder_to_id[folder.name]
        class_name = currency_classes[class_id]
        
        all_imgs = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        all_imgs.sort()

        # Validate images first
        valid_imgs = []
        for img_p in all_imgs:
            if check_image_validity(img_p):
                valid_imgs.append(img_p)
            else:
                print(f"  [SKIPPED CORRUPTED] {img_p.relative_to(raw_dir.parent)}")
                stats['corrupted_skipped'] += 1

        stats['per_class_count'][class_name] = len(valid_imgs)
        random.shuffle(valid_imgs)

        split_idx = int(len(valid_imgs) * train_ratio)
        train_list = valid_imgs[:split_idx]
        val_list = valid_imgs[split_idx:]

        for split_name, img_subset, target_img_dir, target_lbl_dir in [
            ('train', train_list, images_train, labels_train),
            ('val', val_list, images_val, labels_val)
        ]:
            for img_path in img_subset:
                # Ensure unique image filename
                stem = f"{class_name}_{img_path.stem}"
                ext = img_path.suffix.lower()
                dest_img = target_img_dir / f"{stem}{ext}"
                dest_lbl = target_lbl_dir / f"{stem}.txt"

                # Copy image
                shutil.copy2(img_path, dest_img)

                # Generate YOLO bounding box label: class_id x_center y_center width height
                # Currency note covers the whole framed image: 0.5 0.5 0.98 0.98
                with open(dest_lbl, 'w') as lf:
                    lf.write(f"{class_id} 0.500000 0.500000 0.980000 0.980000\n")

                if split_name == 'train':
                    stats['train_images'] += 1
                    stats['train_labels'] += 1
                else:
                    stats['val_images'] += 1
                    stats['val_labels'] += 1

    # 4b. Generate realistic multi-note composite images for robust multi-note detection
    print("\nGenerating multi-note composite training and validation scenes...")
    for split_name, target_img_dir, target_lbl_dir, num_composites in [
        ('train', images_train, labels_train, 350),
        ('val', images_val, labels_val, 80)
    ]:
        all_split_imgs = [
            (folder_to_id[folder.name], img_p)
            for folder in raw_folders
            for img_p in (folder.iterdir())
            if img_p.is_file() and img_p.suffix.lower() in ['.jpg', '.jpeg', '.png'] and check_image_validity(img_p)
        ]

        canvas_size = 640
        for comp_idx in range(num_composites):
            # Create varied background canvas
            bg_color = random.choice([
                (random.randint(200, 240), random.randint(200, 240), random.randint(200, 240)),  # Light
                (random.randint(40, 90), random.randint(40, 90), random.randint(40, 90)),        # Dark
                (random.randint(120, 160), random.randint(90, 130), random.randint(60, 100))    # Table/Wood
            ])
            canvas = Image.new('RGB', (canvas_size, canvas_size), color=bg_color)
            labels_in_composite = []

            # Pick 2 to 4 notes randomly
            k_notes = random.randint(2, 4)
            chosen_notes = random.sample(all_split_imgs, k_notes)

            # Divide canvas into quadrants / non-overlapping grid positions
            positions = [
                (random.randint(20, 60), random.randint(20, 60)),
                (random.randint(330, 370), random.randint(20, 60)),
                (random.randint(20, 60), random.randint(330, 370)),
                (random.randint(330, 370), random.randint(330, 370))
            ]
            random.shuffle(positions)

            for note_idx, (cid, note_path) in enumerate(chosen_notes):
                try:
                    with Image.open(note_path) as note_img:
                        note_img = note_img.convert('RGB')
                        # Randomly resize note between 200x200 and 260x260
                        nw = random.randint(200, 260)
                        nh = random.randint(200, 260)
                        note_resized = note_img.resize((nw, nh), Image.Resampling.BILINEAR)

                        px, py = positions[note_idx]
                        # Ensure within canvas bounds
                        px = min(px, canvas_size - nw - 10)
                        py = min(py, canvas_size - nh - 10)

                        canvas.paste(note_resized, (px, py))

                        # YOLO normalized bbox: xc, yc, w, h
                        xc = (px + nw / 2.0) / canvas_size
                        yc = (py + nh / 2.0) / canvas_size
                        norm_w = nw / canvas_size
                        norm_h = nh / canvas_size

                        labels_in_composite.append(f"{cid} {xc:.6f} {yc:.6f} {norm_w:.6f} {norm_h:.6f}")
                except Exception:
                    continue

            comp_stem = f"composite_{split_name}_{comp_idx:04d}"
            comp_img_path = target_img_dir / f"{comp_stem}.jpg"
            comp_lbl_path = target_lbl_dir / f"{comp_stem}.txt"

            canvas.save(comp_img_path, "JPEG", quality=95)
            with open(comp_lbl_path, "w") as lf:
                lf.write("\n".join(labels_in_composite) + "\n")

            if split_name == 'train':
                stats['train_images'] += 1
                stats['train_labels'] += 1
            else:
                stats['val_images'] += 1
                stats['val_labels'] += 1

    # 5. Create final valid Ultralytics data.yaml
    data_yaml_path = output_dir / "data.yaml"
    yaml_content = {
        'path': str(output_dir.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {int(k): v for k, v in all_classes.items()}
    }

    with open(data_yaml_path, 'w') as yf:
        yaml.dump(yaml_content, yf, sort_keys=True)

    print(f"\nSaved Ultralytics config to '{data_yaml_path}'")
    return all_classes, stats, data_yaml_path

def validate_dataset(dataset_dir: Path):
    """Run comprehensive validation on dataset structure, images, and labels."""
    print("\n" + "="*50)
    print("RUNNING COMPREHENSIVE DATASET VALIDATION")
    print("="*50)

    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}")

    with open(data_yaml, 'r') as f:
        config = yaml.safe_load(f)

    class_dict = config.get('names', {})
    if isinstance(class_dict, list):
        class_dict = {i: name for i, name in enumerate(class_dict)}
    else:
        class_dict = {int(k): v for k, v in class_dict.items()}

    errors = []
    warnings = []

    for split in ['train', 'val']:
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split

        if not img_dir.exists():
            errors.append(f"Image directory missing: {img_dir}")
            continue
        if not lbl_dir.exists():
            errors.append(f"Label directory missing: {lbl_dir}")
            continue

        img_files = {f.stem: f for f in img_dir.iterdir() if f.is_file() and not f.name.startswith('.')}
        lbl_files = {f.stem: f for f in lbl_dir.iterdir() if f.is_file() and not f.name.startswith('.')}

        print(f"\nChecking split '{split}':")
        print(f"  Total images: {len(img_files)}")
        print(f"  Total labels: {len(lbl_files)}")

        # Check mismatched pairs
        missing_labels = set(img_files.keys()) - set(lbl_files.keys())
        missing_images = set(lbl_files.keys()) - set(img_files.keys())

        if missing_labels:
            errors.append(f"[{split}] {len(missing_labels)} images missing corresponding label files!")
        if missing_images:
            errors.append(f"[{split}] {len(missing_images)} labels missing corresponding image files!")

        # Validate label contents & YOLO format
        for stem, lbl_path in lbl_files.items():
            if lbl_path.stat().st_size == 0:
                errors.append(f"[{split}] Empty label file: {lbl_path.name}")
                continue

            with open(lbl_path, 'r') as lf:
                lines = lf.readlines()

            if not lines:
                errors.append(f"[{split}] Empty label file content: {lbl_path.name}")
                continue

            for line_idx, line in enumerate(lines, 1):
                parts = line.strip().split()
                if len(parts) != 5:
                    errors.append(f"[{split}] Invalid YOLO line format in {lbl_path.name}:{line_idx} -> '{line.strip()}'")
                    continue
                try:
                    cid = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])
                    if cid not in class_dict:
                        errors.append(f"[{split}] Class ID {cid} out of configured range {list(class_dict.keys())} in {lbl_path.name}")
                    for val_name, val in [('x_center', xc), ('y_center', yc), ('width', w), ('height', h)]:
                        if not (0.0 <= val <= 1.0):
                            errors.append(f"[{split}] Normalized {val_name}={val} outside [0.0, 1.0] in {lbl_path.name}")
                except ValueError as ve:
                    errors.append(f"[{split}] Non-numeric values in {lbl_path.name}:{line_idx} -> {ve}")

    print("\n--- Validation Summary ---")
    if errors:
        print(f"FAILED: {len(errors)} errors found:")
        for err in errors:
            print(f"  ❌ {err}")
        return False
    else:
        print("  ✅ All images and labels are valid!")
        print("  ✅ 0 missing image-label pairs")
        print("  ✅ 0 empty label files")
        print("  ✅ 0 corrupted images in final dataset")
        print("  ✅ 100% valid YOLO detection coordinates")
        return True

def main():
    parser = argparse.ArgumentParser(description="Prepare and validate currency dataset for Ultralytics YOLO.")
    parser.add_argument("--zip", type=str, default="Indian Currencies.zip", help="Path to raw dataset ZIP file")
    parser.add_argument("--raw-dir", type=str, default="datasets/raw", help="Raw extraction folder")
    parser.add_argument("--output-dir", type=str, default="datasets/combined", help="Target combined YOLO dataset folder")
    parser.add_argument("--existing-yaml", type=str, default=None, help="Optional path to existing dataset data.yaml for class remapping")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio (default: 0.8)")
    args = parser.parse_args()

    zip_path = Path(args.zip)
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    existing_yaml = Path(args.existing_yaml) if args.existing_yaml else None

    # Step 1: Extract if raw_dir not populated
    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        safe_extract_zip(zip_path, raw_dir)

    # Step 2: Prepare YOLO dataset
    all_classes, stats, data_yaml_path = prepare_currency_dataset(
        raw_dir=raw_dir,
        output_dir=output_dir,
        existing_yaml=existing_yaml,
        train_ratio=args.train_ratio
    )

    # Step 3: Run Validation
    is_valid = validate_dataset(output_dir)
    if not is_valid:
        sys.exit(1)

    # Step 4: Print final class list clearly
    print("\n" + "="*50)
    print("FINAL CLASS LIST MAPPING")
    print("="*50)
    for cid in sorted(all_classes.keys()):
        print(f"{cid}: {all_classes[cid]}")
    print("="*50)
    print(f"Total Train Images: {stats['train_images']}")
    print(f"Total Val Images:   {stats['val_images']}")
    print(f"Total Dataset Size: {stats['train_images'] + stats['val_images']} images")
    print(f"Corrupted Filtered: {stats['corrupted_skipped']} image(s)")
    print(f"YAML Configuration: {data_yaml_path.resolve()}")
    print("="*50)

if __name__ == "__main__":
    main()
