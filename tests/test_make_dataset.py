"""
Standard library unittest suite for dataset preparation and stratified splitting.
Runs without requiring pytest or third-party packages.
"""

import os
import shutil
import tempfile
import json
import csv
import unittest
from pathlib import Path

from src.data.make_dataset import (
    DatasetPreparer,
    compute_file_sha256,
    verify_image_integrity,
    DEFAULT_SEED
)


class TestDatasetPreparer(unittest.TestCase):

    def setUp(self):
        # Create a temporary sandbox directory inside the current working directory
        self.workspace_root = Path(__file__).resolve().parent.parent
        self.test_dir = self.workspace_root / "tests" / "temp_test_sandbox"
        self.raw_dir = self.test_dir / "raw"
        self.processed_dir = self.test_dir / "processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        # Clean up temporary test sandbox
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_synthetic_file(self, filepath: Path, color: str = "blue", size=(10, 10), format: str = "JPEG"):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image
            img = Image.new("RGB", size, color=color)
            img.save(filepath, format=format)
        except ImportError:
            with open(filepath, "wb") as f:
                f.write(b"minimal_data")

    def test_missing_raw_dir(self):
        missing_dir = self.raw_dir / "non_existent_folder"
        preparer = DatasetPreparer(raw_dir=missing_dir, processed_dir=self.processed_dir)
        res = preparer.run()
        self.assertEqual(res["status"], "error")
        self.assertIn("does not exist", res["message"])

    def test_empty_raw_dir(self):
        preparer = DatasetPreparer(raw_dir=self.raw_dir, processed_dir=self.processed_dir)
        res = preparer.run()
        self.assertEqual(res["status"], "empty")
        self.assertIn("No artist subdirectories found", res["message"])

    def test_duplicate_detection(self):
        artist_dir = self.raw_dir / "Vincent_van_Gogh"
        img1 = artist_dir / "painting_001.jpg"
        img2 = artist_dir / "painting_002_copy.jpg"
        self._create_synthetic_file(img1, color="red")
        # Copy file exactly to simulate duplicate
        shutil.copyfile(img1, img2)

        img3 = artist_dir / "painting_003.jpg"
        self._create_synthetic_file(img3, color="green")

        preparer = DatasetPreparer(raw_dir=self.raw_dir, processed_dir=self.processed_dir)
        scan = preparer.scan_and_validate()

        self.assertEqual(scan["status"], "success")
        self.assertEqual(len(scan["valid_records"]), 2)  # One duplicate eliminated
        self.assertEqual(len(scan["duplicates"]), 1)
        self.assertEqual(scan["duplicates"][0]["sha256"], compute_file_sha256(img1))

    def test_stratified_split_proportions_and_zero_leakage(self):
        artists = ["Claude_Monet", "Edgar_Degas"]
        images_per_artist = 20

        for a_idx, artist in enumerate(artists):
            a_dir = self.raw_dir / artist
            for i in range(images_per_artist):
                # Generate unique synthetic image by changing pixel colors
                color = ((a_idx * 50 + i * 10) % 255, (i * 12) % 255, (a_idx * 100 + i * 5) % 255)
                self._create_synthetic_file(a_dir / f"work_{i:03d}.jpg", color=color)

        preparer = DatasetPreparer(raw_dir=self.raw_dir, processed_dir=self.processed_dir, seed=42)
        res = preparer.run()

        self.assertEqual(res["status"], "success")
        stats = res["stats"]
        self.assertEqual(stats["total_images"], 40)

        # Check split numbers per artist (20 * 0.8 = 16 train, 2 val, 2 test)
        for artist in artists:
            a_stats = stats["artist_counts"][artist]
            self.assertEqual(a_stats["train"], 16)
            self.assertEqual(a_stats["val"], 2)
            self.assertEqual(a_stats["test"], 2)
            self.assertEqual(a_stats["total"], 20)

        # Verify zero leakage across splits
        with open(res["manifest_json"], "r", encoding="utf-8") as f:
            manifest = json.load(f)

        train_hashes = {r["sha256"] for r in manifest if r["split"] == "train"}
        val_hashes = {r["sha256"] for r in manifest if r["split"] == "val"}
        test_hashes = {r["sha256"] for r in manifest if r["split"] == "test"}

        self.assertEqual(len(train_hashes.intersection(val_hashes)), 0, "Leakage between train and val!")
        self.assertEqual(len(train_hashes.intersection(test_hashes)), 0, "Leakage between train and test!")
        self.assertEqual(len(val_hashes.intersection(test_hashes)), 0, "Leakage between val and test!")

    def test_reproducibility_with_seed(self):
        artist_dir = self.raw_dir / "Rembrandt"
        for i in range(10):
            color = (i * 20, (i * 25) % 255, (i * 30) % 255)
            self._create_synthetic_file(artist_dir / f"r_{i}.jpg", color=color)

        preparer_a = DatasetPreparer(raw_dir=self.raw_dir, processed_dir=self.processed_dir, seed=123)
        records_a = preparer_a.split_records_stratified(preparer_a.scan_and_validate()["valid_records"])

        preparer_b = DatasetPreparer(raw_dir=self.raw_dir, processed_dir=self.processed_dir, seed=123)
        records_b = preparer_b.split_records_stratified(preparer_b.scan_and_validate()["valid_records"])

        self.assertEqual(
            [(r["filename"], r["split"]) for r in records_a],
            [(r["filename"], r["split"]) for r in records_b]
        )

    def test_manifest_files_creation(self):
        self._create_synthetic_file(self.raw_dir / "ArtistA" / "img1.png", color="red", format="PNG")
        self._create_synthetic_file(self.raw_dir / "ArtistA" / "img2.png", color="green", format="PNG")
        self._create_synthetic_file(self.raw_dir / "ArtistA" / "img3.png", color="blue", format="PNG")

        preparer = DatasetPreparer(raw_dir=self.raw_dir, processed_dir=self.processed_dir)
        res = preparer.run()

        self.assertTrue(Path(res["manifest_json"]).exists())
        self.assertTrue(Path(res["manifest_csv"]).exists())
        self.assertTrue(Path(res["summary_json"]).exists())

        with open(res["manifest_csv"], "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 3)
            self.assertIn("artist", rows[0])
            self.assertIn("split", rows[0])
            self.assertIn("rel_path", rows[0])


if __name__ == "__main__":
    unittest.main()
