# ============================================================
# 🛰️ SatLetter Dataset — NASA Landsat Alphabet Downloader
# ============================================================
# Run this cell in Google Colab to download satellite letter images.
# No pip install needed - uses only Python stdlib.
#
# URL pattern (verified April 2026):
#   https://science.nasa.gov/specials/your-name-in-landsat/images/{letter}_{index}.jpg
#
# Verified counts: 71 images total (a:5 b:2 c:3 d:2 e:4 f:2 g:1 h:2
#   i:5 j:3 k:2 l:4 m:3 n:3 o:2 p:2 q:2 r:4 s:3 t:2 u:2 v:4 w:2
#   x:3 y:3 z:2)
# ============================================================

import os
import string
import time
import json
import urllib.request
import urllib.error
from IPython.display import display, HTML, clear_output

BASE_URL = "https://science.nasa.gov/specials/your-name-in-landsat/images/"
OUTPUT_DIR = "dataset/satellite_letters/raw/nasa"

VERIFIED_COUNTS = {
    "a": 5, "b": 2, "c": 3, "d": 2, "e": 4, "f": 2, "g": 1, "h": 2,
    "i": 5, "j": 3, "k": 2, "l": 4, "m": 3, "n": 3, "o": 2, "p": 2,
    "q": 2, "r": 4, "s": 3, "t": 2, "u": 2, "v": 4, "w": 2, "x": 3,
    "y": 3, "z": 2,
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Research/Academic) SatLetter Dataset Builder"
}


def download(url, path, timeout=30):
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            if len(data) > 500:
                open(path, "wb").write(data)
                return True
    except Exception:
        pass
    return False


# Download all images
total_expected = sum(VERIFIED_COUNTS.values())
downloaded = 0
skipped = 0
results = {}

print(f"📡 Downloading {total_expected} NASA Landsat alphabet images...")
print(f"📁 Saving to: {OUTPUT_DIR}\n")

for letter_upper in string.ascii_uppercase:
    letter = letter_upper.lower()
    count = VERIFIED_COUNTS.get(letter, 0)
    letter_dir = os.path.join(OUTPUT_DIR, letter_upper)
    os.makedirs(letter_dir, exist_ok=True)

    got = 0
    for idx in range(count + 3):  # probe a few extra
        fname = f"nasa_landsat_{letter_upper}_{idx:03d}.jpg"
        fpath = os.path.join(letter_dir, fname)
        url = f"{BASE_URL}{letter}_{idx}.jpg"

        if os.path.exists(fpath) and os.path.getsize(fpath) > 500:
            got += 1
            skipped += 1
            continue

        if download(url, fpath):
            got += 1
            downloaded += 1
            time.sleep(0.3)
        else:
            if os.path.exists(fpath) and os.path.getsize(fpath) < 500:
                os.remove(fpath)

    results[letter_upper] = got
    status = "✅" if got > 0 else "❌"
    print(f"  {status} {letter_upper}: {got} images")

# Summary
total = sum(results.values())
print(f"\n{'='*50}")
print(f"✨ Done! {total} images ({downloaded} new, {skipped} cached)")
print(f"📁 Saved to: {OUTPUT_DIR}")

# Save metadata
meta = {"source": "NASA Your Name in Landsat", "base_url": BASE_URL,
        "total": total, "counts": results, "date": time.strftime("%Y-%m-%d")}
open(os.path.join(OUTPUT_DIR, "metadata.json"), "w").write(json.dumps(meta, indent=2))
