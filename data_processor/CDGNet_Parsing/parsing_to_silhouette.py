#!/usr/bin/env python3
"""
Convert CDGNet parsing label images into binary silhouettes
by filtering for specific body regions (Upper Body or Lower Body).

Classes in CDGNet / Gait3D-Parsing:
  0: Background
  1: Head        (Upper Body)
  2: Torso       (Upper Body)
  3: Left-arm    (Upper Body)
  4: Right-arm   (Upper Body)
  5: Left-hand   (Upper Body)
  6: Right-hand  (Upper Body)
  7: Left-leg    (Lower Body)
  8: Right-leg   (Lower Body)
  9: Left-foot   (Lower Body)
  10: Right-foot (Lower Body)
  11: Dress      (Upper / Full Body)
"""

import os
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np
from tqdm import tqdm

PARTS_CONFIG = {
    'upper': [1, 2, 3, 4, 5, 6, 11],  # Head, Torso, Arms, Hands, Dress
    'lower': [7, 8, 9, 10],           # Legs, Feet
    'all': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # Full body silhouette
}


def process_single_image(args_tuple):
    """
    Reads a parsing mask, filters selected class labels, and saves as binary silhouette (0 and 255).
    """
    img_path, input_root, output_root, target_classes = args_tuple

    # 1. Read raw parsing image (grayscale, values in [0, 11])
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False

    # 2. Filter target classes and create binary mask (0 or 255)
    binary_mask = np.isin(img, target_classes).astype(np.uint8) * 255

    # 3. Determine output path maintaining directory structure
    rel_path = img_path.relative_to(input_root)
    out_file_path = output_root / rel_path
    out_file_path.parent.mkdir(parents=True, exist_ok=True)

    # 4. Save as PNG
    cv2.imwrite(str(out_file_path), binary_mask)
    return True


def convert_dataset(input_dir: Path, output_dir: Path, part: str = 'upper', num_workers: int = 8):
    """
    Converts all parsing images in input_dir to silhouettes in output_dir using multiprocessing.
    """
    target_classes = PARTS_CONFIG[part]
    print(f"[*] Processing part: '{part}' | Target class IDs: {target_classes}")
    print(f"[*] Scanning input directory: {input_dir}")

    # Find all PNG/JPG parsing masks
    valid_exts = {'.png', '.jpg', '.jpeg'}
    img_paths = [p for p in input_dir.rglob('*') if p.suffix.lower() in valid_exts]

    if not img_paths:
        print(f"[!] No images found in {input_dir}")
        return

    print(f"[*] Found {len(img_paths)} frames. Starting conversion with {num_workers} workers...")

    tasks = [(p, input_dir, output_dir, target_classes) for p in img_paths]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(process_single_image, tasks), total=len(tasks), desc=f"Extracting {part}-body"))

    print(f"[✓] Done! Silhouettes saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract Upper/Lower body silhouettes from CDGNet parsing results.")
    parser.add_argument("-i", "--input_path", type=str, required=True,
                        help="Path to CDGNet 'Pred_parsing_results' folder.")
    parser.add_argument("-o", "--output_path", type=str, required=True,
                        help="Path where binary silhouette images will be saved.")
    parser.add_argument("-p", "--part", type=str, choices=['upper', 'lower', 'all'], default='upper',
                        help="Body region to extract: 'upper', 'lower', or 'all' (default: upper).")
    parser.add_argument("-n", "--workers", type=int, default=8,
                        help="Number of parallel worker processes (default: 8).")

    args = parser.parse_args()

    input_dir = Path(args.input_path)
    output_dir = Path(args.output_path)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_dir}")

    convert_dataset(input_dir, output_dir, part=args.part, num_workers=args.workers)


if __name__ == '__main__':
    main()
