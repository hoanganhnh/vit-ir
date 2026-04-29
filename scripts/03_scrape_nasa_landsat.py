"""
Script 03: Scrape NASA Landsat Alphabet images
===============================================
Scrapes letter images from NASA's "Your Name in Landsat" tool.
The tool serves satellite images of landforms resembling each letter.
Output: dataset/satellite_letters/raw/nasa/A/ ... Z/
"""

import os
import sys
import string
import time
import json
import hashlib
from pathlib import Path
from urllib.parse import urljoin

import requests
from tqdm import tqdm


# NASA Landsat Letter tool endpoints (may need updating)
NASA_LANDSAT_BASE = "https://svs.gsfc.nasa.gov"
NASA_EARTH_OBS_BASE = "https://earthobservatory.nasa.gov"

# Known NASA Landsat Alphabet image URLs
# These are curated from various NASA sources
# Each letter may have multiple known satellite images
KNOWN_LETTER_SOURCES = {
    # NASA Earth Observatory "Reading the ABCs from Space"
    # These URLs point to known satellite images for each letter
    "base_url": "https://eoimages.gsfc.nasa.gov/images/imagerecords/",
}


def download_image(url: str, save_path: str, timeout: int = 30) -> bool:
    """Download an image from a URL."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Research/Academic Project) SatLetter Dataset Builder"
        }
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "image" not in content_type and "octet-stream" not in content_type:
            return False

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True

    except (requests.RequestException, IOError) as e:
        print(f"  Failed to download {url}: {e}")
        return False


def scrape_landsat_alphabet_gallery(output_dir: str = "dataset/satellite_letters/raw/nasa"):
    """
    Attempt to scrape satellite letter images from NASA sources.

    Strategy:
    1. Try the "Your Name in Landsat" tool's image tiles
    2. Try NASA Earth Observatory gallery
    3. Fall back to generating a guide for manual collection
    """

    print("=" * 60)
    print("NASA Landsat Alphabet Scraper")
    print("=" * 60)

    letters = list(string.ascii_uppercase)

    # Create output directories
    for letter in letters:
        os.makedirs(os.path.join(output_dir, letter), exist_ok=True)

    # Strategy 1: Try "Your Name in Landsat" tile endpoint
    # The tool uses pre-selected Landsat tiles for each letter
    print("\n--- Strategy 1: Your Name in Landsat tiles ---")

    # The Landsat tool typically serves tiles from a known set
    # We try common URL patterns used by NASA tools
    landsat_endpoints = [
        "https://svs.gsfc.nasa.gov/vis/a030000/a030028/landsat_alphabet/{letter}_{idx}.jpg",
        "https://svs.gsfc.nasa.gov/vis/a030000/a030028/images/{letter}_{idx}.jpg",
    ]

    total_downloaded = 0
    letter_counts = {l: 0 for l in letters}

    for letter in tqdm(letters, desc="Trying Landsat tiles"):
        for endpoint_template in landsat_endpoints:
            for idx in range(1, 20):  # Try up to 20 variations
                url = endpoint_template.format(letter=letter.lower(), idx=idx)
                filename = f"nasa_landsat_{letter}_{idx:03d}.jpg"
                save_path = os.path.join(output_dir, letter, filename)

                if os.path.exists(save_path):
                    letter_counts[letter] += 1
                    continue

                if download_image(url, save_path):
                    letter_counts[letter] += 1
                    total_downloaded += 1
                    time.sleep(0.5)  # Rate limiting
                else:
                    # Clean up empty file if created
                    if os.path.exists(save_path) and os.path.getsize(save_path) == 0:
                        os.remove(save_path)

    print(f"\nStrategy 1 results: {total_downloaded} images downloaded")

    # Strategy 2: Try NASA Earth Observatory image records
    print("\n--- Strategy 2: Earth Observatory images ---")

    # Known image record IDs for alphabet letters (from NASA EO)
    # These are approximate - the actual IDs may vary
    eo_image_ids = {
        "A": [84887],
        "B": [84888],
        "C": [84889],
        "D": [84890],
        "E": [84891],
        "F": [84892],
        "G": [84893],
        "H": [84894],
        "I": [84895],
        "J": [84896],
        "K": [84897],
        "L": [84898],
        "M": [84899],
        "N": [84900],
        "O": [84901],
        "P": [84902],
        "Q": [84903],
        "R": [84904],
        "S": [84905],
        "T": [84906],
        "U": [84907],
        "V": [84908],
        "W": [84909],
        "X": [84910],
        "Y": [84911],
        "Z": [84912],
    }

    eo_downloaded = 0
    for letter, ids in tqdm(eo_image_ids.items(), desc="Earth Observatory"):
        for img_id in ids:
            # NASA EO image URL patterns
            for size_suffix in ["", "_lrg"]:
                url = (
                    f"https://eoimages.gsfc.nasa.gov/images/imagerecords/"
                    f"{img_id // 1000 * 1000:05d}/{img_id:05d}/"
                    f"{letter.lower()}_eo{size_suffix}.jpg"
                )
                filename = f"nasa_eo_{letter}_{img_id}{size_suffix}.jpg"
                save_path = os.path.join(output_dir, letter, filename)

                if not os.path.exists(save_path):
                    if download_image(url, save_path):
                        eo_downloaded += 1
                        letter_counts[letter] += 1
                        time.sleep(0.5)
                    else:
                        if os.path.exists(save_path) and os.path.getsize(save_path) == 0:
                            os.remove(save_path)

    print(f"Strategy 2 results: {eo_downloaded} images downloaded")

    # Generate summary and manual collection guide
    print("\n" + "=" * 60)
    print("SCRAPING SUMMARY")
    print("=" * 60)

    total = sum(letter_counts.values())
    print(f"Total images downloaded: {total}")
    print(f"\nPer-letter counts:")
    for letter in letters:
        status = "✓" if letter_counts[letter] > 0 else "✗ NEEDS MANUAL"
        print(f"  {letter}: {letter_counts[letter]} images {status}")

    # Generate manual collection guide
    guide_path = os.path.join(output_dir, "COLLECTION_GUIDE.md")
    generate_collection_guide(guide_path, letter_counts)
    print(f"\nManual collection guide saved to: {guide_path}")

    return letter_counts


def generate_collection_guide(guide_path: str, letter_counts: dict):
    """Generate a guide for manual satellite letter collection."""

    guide = """# Hướng dẫn Thu thập Ảnh Vệ tinh Chữ cái (Manual)

## Mục tiêu
Thu thập 50-100 ảnh vệ tinh cho MỖI chữ cái (A-Z) từ hình dạng tự nhiên/nhân tạo.

## Trạng thái hiện tại
"""
    for letter, count in sorted(letter_counts.items()):
        needed = max(0, 50 - count)
        status = f"✓ {count} ảnh" if count >= 50 else f"⚠️ {count}/50 (cần thêm {needed})"
        guide += f"- **{letter}**: {status}\n"

    guide += """
## Nguồn thu thập

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
|-----|--------------------|--------------------|
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
    scrape_landsat_alphabet_gallery()
