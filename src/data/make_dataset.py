"""
Dataset preparation and stratified splitting pipeline for artist identification.

This module processes raw artist painting collections organized in:
    data/raw/<Artist_Name>/<image_file>

It handles:
    - Discovery and format filtering of images (.jpg, .jpeg, .png, .bmp, .webp)
    - Image integrity and readability verification (using Pillow)
    - Exact duplicate detection using SHA-256 cryptographic hashing
    - Reproducible, leakage-free stratified splitting (80% Train, 10% Val, 10% Test)
    - Generation of portable, relative-path manifests (CSV and JSON)
    - Comprehensive data validation reporting (class counts, split statistics, issues)
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import random

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
DEFAULT_SEED = 42
TARGET_SPLITS = {'train': 0.80, 'val': 0.10, 'test': 0.10}


def compute_file_sha256(filepath: Path, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hex digest of a file to detect exact duplicate images."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_image_integrity(filepath: Path) -> Tuple[bool, Optional[str], Optional[Tuple[int, int]]]:
    """
    Verify that the image file can be opened and parsed cleanly.
    Returns (is_valid, error_message, (width, height)).
    """
    if not PIL_AVAILABLE:
        # Fallback if Pillow is not yet installed: check non-empty file
        if filepath.stat().st_size > 0:
            return True, None, None
        return False, "File is empty (0 bytes)", None

    try:
        with Image.open(filepath) as img:
            img.verify()  # Fast check for file corruption
        # Reopen to get dimensions (verify() closes or invalidates image buffers)
        with Image.open(filepath) as img:
            dimensions = img.size
            _ = img.format
        return True, None, dimensions
    except Exception as e:
        return False, str(e), None


class DatasetPreparer:
    """
    Manages validation, deduplication, and stratified splitting of artist paintings.
    Operates strictly on paths relative to project root or specified workspace.
    """

    def __init__(
        self,
        raw_dir: Path,
        processed_dir: Path,
        seed: int = DEFAULT_SEED,
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # Tolerance check
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-5:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    def scan_and_validate(self) -> Dict[str, Any]:
        """
        Scan raw_dir for artist folders and images, detecting corruptions and duplicates.
        """
        if not self.raw_dir.exists():
            return {
                "status": "error",
                "message": f"Raw data directory does not exist: {self.raw_dir}",
                "valid_records": [],
                "duplicates": [],
                "corrupted": []
            }

        artist_dirs = [d for d in self.raw_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        if not artist_dirs:
            return {
                "status": "empty",
                "message": f"No artist subdirectories found in {self.raw_dir}",
                "valid_records": [],
                "duplicates": [],
                "corrupted": []
            }

        seen_hashes: Dict[str, Path] = {}
        valid_records: List[Dict[str, Any]] = []
        duplicates: List[Dict[str, str]] = []
        corrupted: List[Dict[str, str]] = []

        for artist_dir in sorted(artist_dirs, key=lambda p: p.name):
            artist_name = artist_dir.name
            for file_path in sorted(artist_dir.iterdir(), key=lambda p: p.name):
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue

                # Resolve relative path cleanly
                try:
                    rel_p = str(file_path.relative_to(self.raw_dir.parent.parent)).replace('\\', '/')
                except Exception:
                    rel_p = str(file_path).replace('\\', '/')

                # Check duplicate
                file_hash = compute_file_sha256(file_path)
                if file_hash in seen_hashes:
                    try:
                        orig_rel = str(seen_hashes[file_hash].relative_to(self.raw_dir.parent.parent)).replace('\\', '/')
                    except Exception:
                        orig_rel = str(seen_hashes[file_hash]).replace('\\', '/')
                    duplicates.append({
                        "file": rel_p,
                        "duplicate_of": orig_rel,
                        "artist": artist_name,
                        "sha256": file_hash
                    })
                    continue

                # Check integrity
                is_valid, err, dims = verify_image_integrity(file_path)
                if not is_valid:
                    corrupted.append({
                        "file": rel_p,
                        "artist": artist_name,
                        "error": err or "Unknown corruption"
                    })
                    continue

                # Record seen and valid
                seen_hashes[file_hash] = file_path
                valid_records.append({
                    "artist": artist_name,
                    "rel_path": rel_p,
                    "filename": file_path.name,
                    "sha256": file_hash,
                    "width": dims[0] if dims else None,
                    "height": dims[1] if dims else None
                })

        return {
            "status": "success",
            "message": f"Found {len(valid_records)} valid images across {len(artist_dirs)} artists.",
            "valid_records": valid_records,
            "duplicates": duplicates,
            "corrupted": corrupted
        }

    def split_records_stratified(
        self,
        records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Perform reproducible, leakage-free stratified splitting by artist class.
        Guarantees that each image belongs to exactly one split (train, val, or test).
        """
        rng = random.Random(self.seed)

        # Group records by artist
        by_artist: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in records:
            by_artist[r["artist"]].append(r)

        assigned_records: List[Dict[str, Any]] = []

        for artist, items in sorted(by_artist.items()):
            # Sort items deterministically before shuffling
            sorted_items = sorted(items, key=lambda x: x["sha256"])
            rng.shuffle(sorted_items)

            n_total = len(sorted_items)
            n_train = int(round(n_total * self.train_ratio))
            n_val = int(round(n_total * self.val_ratio))

            # Ensure test gets the remainder to sum exactly to n_total
            # Guard against zero splits if class size is very small
            if n_total >= 3:
                if n_train == 0:
                    n_train = 1
                if n_val == 0:
                    n_val = 1
            n_test = n_total - (n_train + n_val)
            if n_test < 0:
                n_test = 0
                n_val = n_total - n_train

            train_items = sorted_items[:n_train]
            val_items = sorted_items[n_train:n_train + n_val]
            test_items = sorted_items[n_train + n_val:]

            for item in train_items:
                c = dict(item)
                c["split"] = "train"
                assigned_records.append(c)

            for item in val_items:
                c = dict(item)
                c["split"] = "val"
                assigned_records.append(c)

            for item in test_items:
                c = dict(item)
                c["split"] = "test"
                assigned_records.append(c)

        return assigned_records

    def save_manifests(
        self,
        records: List[Dict[str, Any]],
        stats: Dict[str, Any]
    ) -> Tuple[Path, Path, Path]:
        """
        Save dataset split manifests:
            - data/processed/dataset_manifest.json
            - data/processed/dataset_manifest.csv
            - data/processed/dataset_summary.json
        """
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        json_path = self.processed_dir / "dataset_manifest.json"
        csv_path = self.processed_dir / "dataset_manifest.csv"
        summary_path = self.processed_dir / "dataset_summary.json"

        # JSON Manifest
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2)

        # CSV Manifest
        headers = ["artist", "split", "rel_path", "filename", "sha256", "width", "height"]
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write(','.join(headers) + '\n')
            for r in records:
                row = [
                    str(r.get(h, '') if r.get(h) is not None else '')
                    for h in headers
                ]
                f.write(','.join(row) + '\n')

        # Summary
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

        return json_path, csv_path, summary_path

    def run(self) -> Dict[str, Any]:
        """Execute full scan, verification, stratified split, and manifest saving."""
        scan_res = self.scan_and_validate()
        if scan_res["status"] != "success":
            return scan_res

        valid_records = scan_res["valid_records"]
        assigned_records = self.split_records_stratified(valid_records)

        # Compute summary stats
        split_counts: Dict[str, int] = defaultdict(int)
        artist_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for r in assigned_records:
            split_counts[r["split"]] += 1
            artist_counts[r["artist"]][r["split"]] += 1
            artist_counts[r["artist"]]["total"] += 1

        stats = {
            "random_seed": self.seed,
            "target_ratios": {
                "train": self.train_ratio,
                "val": self.val_ratio,
                "test": self.test_ratio
            },
            "total_images": len(assigned_records),
            "split_counts": dict(split_counts),
            "artist_counts": {k: dict(v) for k, v in sorted(artist_counts.items())},
            "duplicates_removed": len(scan_res["duplicates"]),
            "corrupted_skipped": len(scan_res["corrupted"])
        }

        json_p, csv_p, summary_p = self.save_manifests(assigned_records, stats)

        return {
            "status": "success",
            "stats": stats,
            "manifest_json": str(json_p),
            "manifest_csv": str(csv_p),
            "summary_json": str(summary_p),
            "duplicates": scan_res["duplicates"],
            "corrupted": scan_res["corrupted"]
        }


def main():
    """CLI entrypoint for dataset preparation."""
    workspace_root = Path(__file__).resolve().parent.parent.parent
    raw_dir = workspace_root / "data" / "raw"
    processed_dir = workspace_root / "data" / "processed"

    print("=== Artist Identification: Dataset Preparation Pipeline ===")
    print(f"Raw directory:       {raw_dir}")
    print(f"Processed directory: {processed_dir}")

    preparer = DatasetPreparer(raw_dir=raw_dir, processed_dir=processed_dir, seed=DEFAULT_SEED)
    result = preparer.run()

    if result["status"] == "empty":
        print(f"\n[INFO] {result['message']}")
        print("Place artist subfolders in 'data/raw/' before running splitting.")
    elif result["status"] == "error":
        print(f"\n[ERROR] {result['message']}")
    else:
        stats = result["stats"]
        print(f"\n[SUCCESS] Processed {stats['total_images']} images across {len(stats['artist_counts'])} artists.")
        print(f"Split distribution: {stats['split_counts']}")
        print(f"Manifest saved to: {result['manifest_csv']}")


if __name__ == "__main__":
    main()
