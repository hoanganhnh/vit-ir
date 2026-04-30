"""
Script 03: Download NASA Landsat Alphabet images
=================================================
Downloads letter images from NASA's "Your Name in Landsat" tool.
URL pattern: https://science.nasa.gov/specials/your-name-in-landsat/images/{letter}_{index}.jpg

Each letter has multiple satellite images (varying count per letter).
Output: dataset/satellite_letters/raw/nasa/{A..Z}/
"""

import os
import sys
import string
import time
import json
from pathlib import Path

import requests


# NASA "Your Name in Landsat" image endpoint
BASE_URL = "https://science.nasa.gov/specials/your-name-in-landsat/images/"

# Browser-like headers to avoid blocks
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Rate limiting
DELAY_BETWEEN_REQUESTS = 0.3  # seconds


def download_image(url: str, save_path: str, timeout: int = 30) -> bool:
    """Download an image from a URL with retry logic."""
    for attempt in range(3):
        try:
            response = requests.get(url, headers=HEADERS, timeout=timeout)

            if response.status_code == 200:
                # Verify it's actually an image (check content-type or file magic)
                content_type = response.headers.get("content-type", "")
                if "image" in content_type or "octet-stream" in content_type:
                    with open(save_path, "wb") as f:
                        f.write(response.content)
                    return True
                # Sometimes NASA returns HTML for missing images with 200
                if len(response.content) < 1000:
                    return False
                # Save anyway if content is large enough to be an image
                with open(save_path, "wb") as f:
                    f.write(response.content)
                return True

            elif response.status_code == 404:
                return False

            elif response.status_code == 429:
                # Rate limited - back off
                wait = 5 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue

            else:
                return False

        except requests.exceptions.Timeout:
            if attempt < 2:
                time.sleep(2)
                continue
            return False
        except requests.RequestException as e:
            print(f"  Request error: {e}")
            return False

    return False


def scrape_landsat_alphabet(
    output_dir: str = "dataset/satellite_letters/raw/nasa",
    max_per_letter: int = 50,
):
    """
    Download satellite letter images from NASA's Your Name in Landsat tool.

    The tool serves images at:
        https://science.nasa.gov/specials/your-name-in-landsat/images/{letter}_{index}.jpg

    where letter is a-z (lowercase) and index starts at 0.
    We iterate until we get a 404 (no more images for that letter).
    """

    print("=" * 60)
    print("NASA Landsat Alphabet Downloader")
    print("=" * 60)
    print(f"Source: {BASE_URL}")
    print(f"Output: {output_dir}")
    print(f"Max per letter: {max_per_letter}")
    print()

    letters = list(string.ascii_uppercase)

    # Create output directories (uppercase folder names to match project convention)
    for letter in letters:
        os.makedirs(os.path.join(output_dir, letter), exist_ok=True)

    # Track results
    letter_counts = {letter: 0 for letter in letters}
    total_downloaded = 0
    total_skipped = 0

    for li, letter in enumerate(letters):
        print(f"[{li + 1:2d}/26] Downloading letter {letter}...", end=" ", flush=True)
        letter_lower = letter.lower()
        index = 0
        consecutive_failures = 0

        while index < max_per_letter:
            # NASA uses lowercase letter in URL
            filename_remote = f"{letter_lower}_{index}.jpg"
            url = f"{BASE_URL}{filename_remote}"

            # Save with uppercase letter folder, descriptive filename
            filename_local = f"nasa_landsat_{letter}_{index:03d}.jpg"
            save_path = os.path.join(output_dir, letter, filename_local)

            # Skip if already downloaded
            if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                letter_counts[letter] += 1
                total_skipped += 1
                index += 1
                consecutive_failures = 0
                continue

            if download_image(url, save_path):
                letter_counts[letter] += 1
                total_downloaded += 1
                consecutive_failures = 0
                time.sleep(DELAY_BETWEEN_REQUESTS)
            else:
                # Clean up empty/invalid files
                if os.path.exists(save_path):
                    if os.path.getsize(save_path) == 0:
                        os.remove(save_path)
                consecutive_failures += 1
                # After 2 consecutive failures, assume no more images for this letter
                if consecutive_failures >= 2:
                    break

            index += 1

        print(f"{letter_counts[letter]} images")

    # Print summary
    print()
    print("=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    total_images = sum(letter_counts.values())
    print(f"Total images: {total_images}")
    print(f"  New downloads: {total_downloaded}")
    print(f"  Already existed: {total_skipped}")
    print()

    # Per-letter breakdown
    print("Per-letter counts:")
    empty_letters = []
    for letter in letters:
        count = letter_counts[letter]
        if count == 0:
            empty_letters.append(letter)
            status = "✗ EMPTY"
        else:
            status = "✓"
        print(f"  {letter}: {count:3d} images {status}")

    if empty_letters:
        print(f"\n⚠️  Letters with no images: {', '.join(empty_letters)}")
        print("  These may need manual collection from Google Earth.")

    # Save metadata
    metadata = {
        "source": "NASA Your Name in Landsat",
        "base_url": BASE_URL,
        "total_images": total_images,
        "letter_counts": letter_counts,
    }
    metadata_path = os.path.join(output_dir, "download_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to: {metadata_path}")

    # Generate collection guide for missing letters
    guide_path = os.path.join(output_dir, "COLLECTION_GUIDE.md")
    generate_collection_guide(guide_path, letter_counts)
    print(f"Collection guide: {guide_path}")

    return letter_counts


def generate_collection_guide(guide_path: str, letter_counts: dict):
    """Generate a guide for manual satellite letter collection."""

    guide = """# Hướng dẫn Thu thập Ảnh Vệ tinh Chữ cái (Manual)

## Mục tiêu
Thu thập 50-100 ảnh vệ tinh cho MỖI chữ cái (A-Z) từ hình dạng tự nhiên/nhân tạo.

## Trạng thái hiện tại (từ NASA scraper)
"""
    for letter, count in sorted(letter_counts.items()):
        needed = max(0, 50 - count)
        status = f"✓ {count} ảnh" if count >= 50 else f"⚠️ {count}/50 (cần thêm {needed})"
        guide += f"- **{letter}**: {status}\n"

    guide += """
## Nguồn thu thập bổ sung

### 1. Google Earth Pro (FREE)
- Download: https://www.google.com/earth/versions/#earth-pro
- Mở Google Earth → Tìm các hình dạng tự nhiên → Screenshot → Crop 224x224

### 2. NASA Worldview
- URL: https://worldview.earthdata.nasa.gov/
- Tìm kiếm theo khu vực → Screenshot

### 3. Sentinel Hub EO Browser
- URL: https://apps.sentinel-hub.com/eo-browser/
- Free tier, ảnh Sentinel-2 resolution ~10m

## Gợi ý hình dạng cho từng chữ cái

| Chữ | Hình dạng tự nhiên | Hình dạng nhân tạo |
|-----|--------------------|---------------------|
| A | Núi nhọn, tam giác delta | Tháp, cầu dây |
| B | Hai hồ nước cạnh nhau | Sân vận động đôi |
| C | Vịnh cong, bờ biển cong | Đường cong cao tốc, dam |
| D | Hồ hình bán nguyệt | Sân bay, nhà kho |
| E | Nhánh sông song song | Bến cảng, đường sắt |
| F | Nhánh sông chữ F | Giao lộ, cấu trúc nhà máy |
| G | Bờ biển vòng cung | Bến tàu vòng cung |
| H | Hai con sông + cầu nối | Đường giao nhau, cánh đồng |
| I | Sông thẳng, khe núi | Đường cao tốc thẳng, đập |
| J | Sông uốn cong 1 đầu | Bến cảng hình móc |
| K | Nhánh sông phân kỳ | Giao lộ phức tạp |
| L | Sông gấp khúc 90° | Góc đường, bờ kè |
| M | Dãy núi zigzag | Cầu dây văng |
| N | Sông zigzag | Đường núi ngoằn ngoèo |
| O | Hồ tròn, miệng núi lửa | Sân vận động, đảo san hô |
| P | Bán đảo + cổ đất | Bến cảng chữ P |
| Q | Hồ tròn + suối chảy ra | Sân bay vòng xoay |
| R | Sông phân nhánh | Đường phân nhánh |
| S | Sông uốn khúc | Đường serpentine |
| T | Hợp lưu sông chữ T | Giao lộ chữ T, đường băng |
| U | Thung lũng hình chữ U | Đập nước, sân vận động |
| V | Thung lũng hẹp, delta ngược | Đường giao nhau góc nhọn |
| W | Bờ biển zigzag | Đường núi, nhà kho |
| X | Giao điểm hai sông | Giao lộ, đường sắt |
| Y | Hợp lưu ba nhánh sông | Ngã ba đường |
| Z | Sông zigzag hình Z | Đường zigzag núi |

## Quy tắc khi thu thập
1. **Kích thước**: Crop ảnh về 224×224 pixels hoặc tỉ lệ 1:1
2. **Chất lượng**: Ảnh rõ ràng, hình dạng chữ cái dễ nhận ra
3. **Đa dạng**: Thu thập từ nhiều vùng địa lý, nhiều loại hình dạng
4. **Đặt tên**: `{nguồn}_{chữ_cái}_{số_thứ_tự}.jpg`
   - Ví dụ: `ge_S_001.jpg` (Google Earth, chữ S, ảnh thứ 1)
5. **Lưu vào**: `dataset/satellite_letters/raw/google_earth/{chữ_cái}/`
"""
    with open(guide_path, "w", encoding="utf-8") as f:
        f.write(guide)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download NASA Landsat alphabet images")
    parser.add_argument(
        "--output", "-o",
        default="dataset/satellite_letters/raw/nasa",
        help="Output directory (default: dataset/satellite_letters/raw/nasa)",
    )
    parser.add_argument(
        "--max-per-letter", "-m",
        type=int,
        default=50,
        help="Max images to try per letter (default: 50)",
    )
    args = parser.parse_args()

    scrape_landsat_alphabet(output_dir=args.output, max_per_letter=args.max_per_letter)
