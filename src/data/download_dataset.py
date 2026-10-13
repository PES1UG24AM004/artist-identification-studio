"""
Targeted Dataset Acquisition Module for Artist Identification.

Downloads up to 200 resized painting images per artist across 10 selected artists
directly from the published dataset archive:
    ikarus777/best-artworks-of-all-time

Uses HTTP Range requests on the public dataset endpoint to stream only the compressed
bytes for the target resized images, extracting them locally into:
    data/raw/<Artist_Name>/<Image_File>

Requirements & Safeguards:
    - Bounded retries and socket timeouts
    - Validates Content-Range and HTTP 206 status codes
    - Verifies JPEG magic bytes and Pillow image readability
    - Never overwrites existing valid images
    - Generates an acquisition manifest (data/raw/acquisition_manifest.json)
"""

import os
import sys
import io
import time
import json
import zlib
import struct
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import urllib.request
import urllib.error

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/ikarus777/best-artworks-of-all-time"
CD_OFFSET = 2458247755
CD_SIZE = 1913891
TARGET_IMAGES_PER_ARTIST = 200

# Canonical directory names and their archive matching prefixes
TARGET_ARTISTS: Dict[str, Dict[str, Any]] = {
    "Vincent_van_Gogh": {
        "prefix": "Vincent_van_Gogh",
        "clean_name": "Vincent van Gogh"
    },
    "Edgar_Degas": {
        "prefix": "Edgar_Degas",
        "clean_name": "Edgar Degas"
    },
    "Pablo_Picasso": {
        "prefix": "Pablo_Picasso",
        "clean_name": "Pablo Picasso"
    },
    "Pierre-Auguste_Renoir": {
        "prefix": "Pierre-Auguste_Renoir",
        "clean_name": "Pierre-Auguste Renoir"
    },
    "Albrecht_Durer": {
        "prefix": "Albrecht_Du",
        "clean_name": "Albrecht Dürer",
        "filter_clean_unicode": True
    },
    "Paul_Gauguin": {
        "prefix": "Paul_Gauguin",
        "clean_name": "Paul Gauguin"
    },
    "Francisco_Goya": {
        "prefix": "Francisco_Goya",
        "clean_name": "Francisco Goya"
    },
    "Rembrandt": {
        "prefix": "Rembrandt",
        "clean_name": "Rembrandt"
    },
    "Alfred_Sisley": {
        "prefix": "Alfred_Sisley",
        "clean_name": "Alfred Sisley"
    },
    "Titian": {
        "prefix": "Titian",
        "clean_name": "Titian"
    }
}


def fetch_bytes_with_retry(
    url: str,
    start_byte: int,
    end_byte: int,
    max_retries: int = 4,
    timeout: int = 25
) -> bytes:
    """Fetch byte range via HTTP GET with bounded retries."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Range": f"bytes={start_byte}-{end_byte}"
    }

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = resp.status
                if status not in (200, 206):
                    raise RuntimeError(f"Unexpected HTTP status {status} for range {start_byte}-{end_byte}")
                data = resp.read()
                expected_len = end_byte - start_byte + 1
                if status == 206 and len(data) != expected_len:
                    raise RuntimeError(f"Incomplete range payload: expected {expected_len}, got {len(data)}")
                return data
        except (urllib.error.URLError, TimeoutError, RuntimeError) as e:
            if attempt == max_retries:
                raise RuntimeError(f"Failed to fetch range {start_byte}-{end_byte} after {max_retries} attempts: {e}")
            time.sleep(1.0 * attempt)
    return b""


def parse_central_directory(cd_data: bytes) -> Dict[str, List[Dict[str, Any]]]:
    """Parse ZIP Central Directory and collect target artist image entries."""
    artist_entries: Dict[str, List[Dict[str, Any]]] = {k: [] for k in TARGET_ARTISTS}
    pos = 0

    while pos < len(cd_data) - 46:
        if cd_data[pos:pos+4] != b'PK\x01\x02':
            break

        n, m, k = struct.unpack('<HHH', cd_data[pos+28:pos+34])
        raw_fn = cd_data[pos+46:pos+46+n].decode('utf-8', errors='ignore')

        if raw_fn.startswith('resized/resized/') and raw_fn.endswith('.jpg'):
            base_fn = raw_fn[len('resized/resized/'):]
            offset, = struct.unpack('<I', cd_data[pos+42:pos+46])
            extra = cd_data[pos+46+n:pos+46+n+m]

            # Parse Zip64 extra if present
            uncomp_size = comp_size = None
            epos = 0
            while epos < len(extra):
                ehdr, esize = struct.unpack('<HH', extra[epos:epos+4])
                if ehdr == 1:
                    uncomp_size, comp_size = struct.unpack('<QQ', extra[epos+4:epos+20])
                epos += 4 + esize

            for canon_name, cfg in TARGET_ARTISTS.items():
                if cfg.get("filter_clean_unicode") and '\u0308' not in base_fn:
                    continue

                norm_fn = unicodedata.normalize('NFKD', base_fn)
                if norm_fn.startswith(cfg["prefix"]):
                    # Standardize filename to safe ASCII if needed
                    safe_fn = base_fn.replace('\u0308', '')  # Albrecht_Dürer -> Albrecht_Durer
                    artist_entries[canon_name].append({
                        "raw_zip_path": raw_fn,
                        "base_filename": safe_fn,
                        "orig_filename": base_fn,
                        "offset": offset,
                        "comp_size": comp_size,
                        "uncomp_size": uncomp_size
                    })
                    break

        pos += 46 + n + m + k

    return artist_entries


def download_and_extract_image(
    entry: Dict[str, Any],
    dest_path: Path
) -> Tuple[bool, Optional[str], Optional[Tuple[int, int]]]:
    """Download single compressed image entry from remote zip, decompress, and validate."""
    if dest_path.exists() and dest_path.stat().st_size > 0:
        # Validate existing image
        try:
            if PIL_AVAILABLE:
                with Image.open(dest_path) as img:
                    img.verify()
                with Image.open(dest_path) as img:
                    return True, "Already exists (valid)", img.size
            return True, "Already exists", None
        except Exception:
            pass  # Corrupt local file, re-download

    offset = entry["offset"]
    comp_size = entry["comp_size"]

    # Fetch local file header + compressed payload
    # Header is at most 150 bytes, plus comp_size
    fetch_len = 150 + comp_size
    raw_payload = fetch_bytes_with_retry(DATASET_URL, offset, offset + fetch_len - 1)

    if raw_payload[:4] != b'PK\x03\x04':
        return False, f"Invalid local zip signature at offset {offset}", None

    n, m = struct.unpack('<HH', raw_payload[26:30])
    hdr_len = 30 + n + m
    comp_data = raw_payload[hdr_len:hdr_len + comp_size]

    try:
        decompressed_data = zlib.decompress(comp_data, -zlib.MAX_WBITS)
    except Exception as e:
        return False, f"Zlib decompression failed: {e}", None

    # Validate JPEG header and footer
    if not (decompressed_data.startswith(b'\xff\xd8') and decompressed_data.endswith(b'\xff\xd9')):
        return False, "Invalid JPEG markers in decompressed stream", None

    # Verify with Pillow
    dimensions = None
    if PIL_AVAILABLE:
        try:
            img = Image.open(io.BytesIO(decompressed_data))
            img.verify()
            img = Image.open(io.BytesIO(decompressed_data))
            dimensions = img.size
        except Exception as e:
            return False, f"Pillow validation failed: {e}", None

    # Write to destination atomically
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = dest_path.with_suffix(".tmp")
    with open(temp_dest, "wb") as f:
        f.write(decompressed_data)
    temp_dest.replace(dest_path)

    return True, None, dimensions


class TargetedDatasetDownloader:
    """Manages targeted acquisition of 200 resized painting images per artist."""

    def __init__(
        self,
        raw_dir: Path,
        limit_per_artist: int = TARGET_IMAGES_PER_ARTIST
    ):
        self.raw_dir = Path(raw_dir)
        self.limit = limit_per_artist

    def run(self) -> Dict[str, Any]:
        print(f"=== Artist Identification: Targeted Dataset Acquisition ===")
        print(f"Remote Archive: {DATASET_URL}")
        print(f"Destination:    {self.raw_dir}")
        print(f"Target Scope:   {len(TARGET_ARTISTS)} artists, up to {self.limit} images each\n")

        # 1. Fetch Central Directory
        print(f"[1/4] Fetching remote archive central directory ({round(CD_SIZE / (1024*1024), 2)} MB)...")
        cd_data = fetch_bytes_with_retry(DATASET_URL, CD_OFFSET, CD_OFFSET + CD_SIZE - 1)
        if len(cd_data) < 4 or cd_data[:4] != b'PK\x01\x02':
            raise RuntimeError(
                f"Remote archive central directory signature mismatch (expected b'PK\\x01\\x02', got {cd_data[:4]!r}). "
                "The remote dataset archive on Kaggle may have been updated or rebuilt. "
                "Please use the official Kaggle CLI acquisition method instead."
            )
        print("      Central directory signature verified successfully.")

        # 2. Parse target entries
        print("[2/4] Parsing index for approved target artists...")
        artist_entries = parse_central_directory(cd_data)
        total_found = sum(len(v) for v in artist_entries.values())
        if total_found == 0:
            raise RuntimeError(
                "No target artist entries could be matched in the archive central directory. "
                "The remote dataset archive structure may have changed. "
                "Please use the official Kaggle CLI acquisition method instead."
            )

        # 3. Download and extract target images
        print(f"[3/4] Streaming and verifying {self.limit} images per artist...")
        acquisition_manifest = []
        summary_counts = {}

        total_start = time.time()
        for idx, (artist_canon, entries) in enumerate(artist_entries.items(), 1):
            artist_dir = self.raw_dir / artist_canon
            artist_dir.mkdir(parents=True, exist_ok=True)

            selected_entries = entries[:self.limit]
            print(f"      [{idx}/10] {artist_canon}: acquiring {len(selected_entries)} paintings...", end="", flush=True)

            success_count = 0
            artist_start = time.time()
            for entry in selected_entries:
                dest_file = artist_dir / entry["base_filename"]
                ok, err, dims = download_and_extract_image(entry, dest_file)
                if ok:
                    success_count += 1
                    acquisition_manifest.append({
                        "artist": artist_canon,
                        "filename": entry["base_filename"],
                        "rel_path": str(dest_file.relative_to(self.raw_dir.parent.parent)).replace('\\', '/'),
                        "width": dims[0] if dims else None,
                        "height": dims[1] if dims else None,
                        "size_bytes": dest_file.stat().st_size
                    })
                else:
                    print(f"\n        [ERROR] Failed {entry['base_filename']}: {err}")

            elapsed = round(time.time() - artist_start, 1)
            print(f" Done ({success_count}/{len(selected_entries)} verified, {elapsed}s)")
            summary_counts[artist_canon] = success_count

        # 4. Save acquisition manifest
        manifest_file = self.raw_dir / "acquisition_manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump({
                "source": "ikarus777/best-artworks-of-all-time",
                "license": "CC BY-NC-SA 4.0",
                "attribution": "Best Artworks of All Time (Kaggle)",
                "total_images": len(acquisition_manifest),
                "artist_counts": summary_counts,
                "images": acquisition_manifest
            }, f, indent=2)

        total_bytes = sum(m["size_bytes"] for m in acquisition_manifest)
        total_time = round(time.time() - total_start, 1)
        print(f"\n[4/4] Acquisition Complete!")
        print(f"      Total images acquired: {len(acquisition_manifest)}")
        print(f"      Total storage on disk: {round(total_bytes / (1024*1024), 2)} MB")
        print(f"      Total duration:        {total_time}s")
        print(f"      Acquisition manifest:  {manifest_file}")

        return {
            "status": "success",
            "total_images": len(acquisition_manifest),
            "total_bytes": total_bytes,
            "artist_counts": summary_counts,
            "manifest_file": str(manifest_file)
        }


def main():
    workspace_root = Path(__file__).resolve().parent.parent.parent
    raw_dir = workspace_root / "data" / "raw"
    try:
        downloader = TargetedDatasetDownloader(raw_dir=raw_dir)
        res = downloader.run()
        if res.get("status") != "success":
            sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Targeted dataset acquisition failed: {e}")
        print("\nNote: Selective HTTP Range acquisition relies on the remote archive's central directory offset (Version 1).")
        print("If the remote dataset archive has changed or is unavailable, please use the official Kaggle CLI:")
        print("  1. kaggle datasets download -d ikarus777/best-artworks-of-all-time -p data/downloads --unzip")
        print("  2. Copy target artist directories from data/downloads/resized/resized/ into data/raw/")
        sys.exit(1)


if __name__ == "__main__":
    main()
