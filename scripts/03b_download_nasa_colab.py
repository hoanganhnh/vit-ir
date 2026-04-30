"""
Script 03b: Download NASA Landsat Alphabet images (Colab-ready)
================================================================
Run this in Google Colab or any environment with internet access.

Verified URL pattern:
  https://science.nasa.gov/specials/your-name-in-landsat/images/{letter}_{index}.jpg

Verified letter counts (71 images total):
  a:5 b:2 c:3 d:2 e:4 f:2 g:1 h:2 i:5 j:3 k:2 l:4 m:3
  n:3 o:2 p:2 q:2 r:4 s:3 t:2 u:2 v:4 w:2 x:3 y:3 z:2

Usage (Colab):
  !python scripts/03b_download_nasa_colab.py

Usage (local):
  python scripts/03b_download_nasa_colab.py --output /path/to/output
"""

import os
import string
import time
import json
import urllib.request
import urllib.error
from pathlib import Path


# NASA "Your Name in Landsat" image endpoint (verified April 2026)
BASE_URL = "https://science.nasa.gov/specials/your-name-in-landsat/images/"

# Verified image counts per letter (from JS HEAD request testing)
VERIFIED_COUNTS = {
    "a": 5, "b": 2, "c": 3, "d": 2, "e": 4, "f": 2, "g": 1, "h": 2,
    "i": 5, "j": 3, "k": 2, "l": 4, "m": 3, "n": 3, "o": 2, "p": 2,
    "q": 2, "r": 4, "s": 3, "t": 2, "u": 2, "v": 4, "w": 2, "x": 3,
    "y": 3, "z": 2,
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def download_image(url: str, save_path: str, timeout: int = 30) -> bool:
    """Download an image using urllib (no pip dependencies needed)."""
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status == 200:
                    data = response.read()
                    if len(data) > 1000:  # Minimum size to be a valid image
                        with open(save_path, "wb") as f:
                            f.write(data)
                        return True
            return False
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            if e.code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            return False
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt < 2:
                time.sleep(2)
                continue
            return False
    return False


def download_all(
    output_dir: str = "dataset/satellite_letters/raw/nasa",
    probe_extra: bool = True,
):
    """
    Download all known NASA Landsat alphabet images.

    Args:
        output_dir: Where to save images ({output_dir}/{A..Z}/)
        probe_extra: If True, try indices beyond verified counts to find new images
    """
    print("=" * 60)
    print("NASA Landsat Alphabet Downloader")
    print("=" * 60)
    print(f"Source: {BASE_URL}")
    print(f"Output: {output_dir}")
    print(f"Expected: {sum(VERIFIED_COUNTS.values())} images (71 total)")
    print()

    letters = list(string.ascii_uppercase)
    for letter in letters:
        os.makedirs(os.path.join(output_dir, letter), exist_ok=True)

    letter_counts = {}
    total_downloaded = 0
    total_skipped = 0

    for letter_upper in letters:
        letter_lower = letter_upper.lower()
        max_idx = VERIFIED_COUNTS.get(letter_lower, 0)

        # Optionally probe a few extra indices
        if probe_extra:
            max_idx += 3

        print(f"[{letter_upper}] ", end="", flush=True)
        count = 0

        for idx in range(max_idx):
            url = f"{BASE_URL}{letter_lower}_{idx}.jpg"
            filename = f"nasa_landsat_{letter_upper}_{idx:03d}.jpg"
            save_path = os.path.join(output_dir, letter_upper, filename)

            if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
                count += 1
                total_skipped += 1
                print(".", end="", flush=True)
                continue

            if download_image(url, save_path):
                count += 1
                total_downloaded += 1
                print("✓", end="", flush=True)
                time.sleep(0.3)
            else:
                if os.path.exists(save_path) and os.path.getsize(save_path) < 1000:
                    os.remove(save_path)
                print("✗", end="", flush=True)

        letter_counts[letter_upper] = count
        print(f" → {count} images")

    # Summary
    total = sum(letter_counts.values())
    print()
    print("=" * 60)
    print(f"DONE! {total} images ({total_downloaded} new, {total_skipped} cached)")
    print("=" * 60)

    for letter_upper in letters:
        c = letter_counts.get(letter_upper, 0)
        status = "✓" if c > 0 else "✗ EMPTY"
        print(f"  {letter_upper}: {c:3d} {status}")

    # Save metadata
    metadata = {
        "source": "NASA Your Name in Landsat",
        "base_url": BASE_URL,
        "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_images": total,
        "letter_counts": letter_counts,
    }
    meta_path = os.path.join(output_dir, "download_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return letter_counts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download NASA Landsat alphabet images (no pip deps)"
    )
    parser.add_argument(
        "--output", "-o",
        default="dataset/satellite_letters/raw/nasa",
        help="Output directory",
    )
    parser.add_argument(
        "--no-probe",
        action="store_true",
        help="Don't probe for extra images beyond verified counts",
    )
    args = parser.parse_args()

    download_all(output_dir=args.output, probe_extra=not args.no_probe)
