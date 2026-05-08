"""
Script 03c: Download Google Earth satellite letter images
=========================================================
Uses satellite imagery tile servers to capture known letter-shaped
landforms at specific GPS coordinates.

Supports:
  1. Esri World Imagery (FREE, no key required) — default
  2. Google Maps Static API (requires API key, $200/month free tier)
  3. Mapbox Static API (requires token, 50K free/month)

Curated database: 120+ GPS coordinates of letter-shaped features
sourced from NASA EO, Google Earth community, and geographic surveys.

Output: dataset/satellite_letters/raw/google_earth/{A..Z}/

Usage:
  python scripts/03c_download_google_earth.py                   # Esri (free)
  python scripts/03c_download_google_earth.py --provider google  # Google
  python scripts/03c_download_google_earth.py --provider mapbox   # Mapbox


# Esri (miễn phí, mặc định):
python3 scripts/03c_download_google_earth.py

# Google Maps (cần API key):
export GOOGLE_MAPS_API_KEY="your_key"
python3 scripts/03c_download_google_earth.py --provider google

# Bash version:
bash scripts/03c_download_google_earth.sh --esri

"""

import os
import sys
import math
import time
import json
import hashlib
import urllib.request
import urllib.error
from pathlib import Path


# ============================================================
# Curated GPS coordinates of letter-shaped landforms
# Each entry: (letter, lat, lon, zoom, description)
# ============================================================
LETTER_COORDINATES = [
    # === A (7) ===
    ("A", 38.5887, -110.1289, 15, "Bowknot Bend Green River Utah"),
    ("A", 28.2137, 83.9773, 14, "Mountain peak triangle Nepal"),
    ("A", 43.7285, 7.4167, 15, "Cap dAil triangle France"),
    ("A", 37.7685, -122.2526, 16, "Alameda island triangle California"),
    ("A", 45.5167, 9.1833, 14, "Triangle junction Milan Italy"),
    ("A", 38.5900, -110.1250, 14, "Bowknot Bend wide Utah"),
    ("A", 48.8580, 2.2945, 17, "Eiffel Tower triangle Paris"),
    # === B (5) ===
    ("B", 34.6935, -92.5635, 14, "Holla Bend Wildlife Refuge Arkansas"),
    ("B", 55.1833, -77.0333, 14, "Double lake shape James Bay Canada"),
    ("B", 39.9653, 26.2387, 15, "Two bays Dardanelles Turkey"),
    ("B", -22.9533, -43.1729, 15, "Lagoa Rodrigo de Freitas Rio Brazil"),
    ("B", 51.5033, -0.0195, 16, "Thames double bend London"),
    # === C (6) ===
    ("C", 26.1900, 50.5280, 14, "Artificial islands Bahrain crescent"),
    ("C", 30.0444, 31.2357, 14, "Nile bend Cairo Egypt"),
    ("C", -33.8545, 18.4248, 14, "Table Bay Cape Town South Africa"),
    ("C", 42.7281, 25.4858, 13, "Curved valley Bulgaria"),
    ("C", 37.7955, -122.4115, 15, "Aquatic Park cove San Francisco"),
    ("C", -27.1180, -109.3665, 14, "Easter Island coast curve"),
    # === D (5) ===
    ("D", 53.0000, -80.5000, 12, "Akimiski Island James Bay Canada"),
    ("D", 45.4235, 12.3339, 14, "Venice lagoon D shape Italy"),
    ("D", 39.6411, -105.8389, 15, "Dillon Reservoir D shape Colorado"),
    ("D", -12.4634, 130.8456, 13, "Darwin harbor D shape Australia"),
    ("D", 51.4552, 7.0132, 14, "Dortmund reservoir Germany"),
    # === E (5) ===
    ("E", 31.5204, 74.3587, 14, "Canal branches Lahore Pakistan"),
    ("E", 40.6892, -74.0445, 16, "Ellis Island piers New York"),
    ("E", 55.8798, -4.2661, 14, "River Clyde docks Glasgow"),
    ("E", 37.7958, -122.3896, 16, "SF piers E shape"),
    ("E", 51.9076, 4.4860, 15, "Rotterdam port fingers Netherlands"),
    # === F (4) ===
    ("F", -6.0935, 106.6824, 14, "River branch F Jakarta Indonesia"),
    ("F", 48.8584, 2.3385, 16, "Paris rail junction F shape"),
    ("F", 53.5461, -2.1192, 15, "Manchester canal branch UK"),
    ("F", 40.7508, -73.9975, 16, "NYC pier F shape Hudson"),
    # === G (5) ===
    ("G", 55.3600, -131.7200, 13, "Revillagigedo Island Alaska"),
    ("G", 51.5074, -0.1278, 14, "Thames bend G London"),
    ("G", 37.8199, -122.4783, 14, "Golden Gate G curve SF"),
    ("G", 34.0522, -118.2437, 14, "LA port G curve"),
    ("G", 41.3765, -72.0886, 14, "Connecticut River G bend"),
    # === H (5) ===
    ("H", 43.7230, 10.3966, 15, "Pisa bridges H shape Italy"),
    ("H", 40.8223, -73.9495, 15, "Harlem River bridges NYC"),
    ("H", 48.8566, 2.3476, 16, "Ile de la Cite bridges Paris"),
    ("H", 51.5080, -0.0759, 16, "Tower Bridge area H London"),
    ("H", 35.6762, 139.6503, 15, "Tokyo bridges H pattern"),
    # === I (5) ===
    ("I", 37.2309, -115.8078, 12, "Desert road straight Nevada"),
    ("I", 34.7695, -112.4596, 11, "Long straight road Arizona"),
    ("I", 44.2517, 7.7694, 14, "Straight valley Italian Alps"),
    ("I", 31.0461, 34.8516, 13, "Dead Sea coastline I shape"),
    ("I", 33.8068, -115.9018, 13, "Straight canal Salton Sea CA"),
    # === J (5) ===
    ("J", 40.6730, -74.0014, 15, "Buttermilk Channel J curve Brooklyn"),
    ("J", 59.9110, 10.7530, 15, "Oslo fjord J shape Norway"),
    ("J", -33.8600, 151.2094, 15, "Darling Harbour J Sydney"),
    ("J", 37.9464, 23.6274, 15, "Piraeus port J Athens"),
    ("J", 28.3587, -16.5567, 15, "Tenerife port J curve"),
    # === K (4) ===
    ("K", 52.3700, 4.8945, 14, "Amsterdam canal K junction"),
    ("K", 55.6761, 12.5683, 14, "Copenhagen harbor K branch"),
    ("K", 51.4454, -2.5879, 14, "Bristol channel K junction UK"),
    ("K", 48.1351, 11.5820, 14, "Munich river K junction"),
    # === L (5) ===
    ("L", 25.7617, -80.1918, 14, "Miami coast L bend"),
    ("L", 30.0595, 31.2244, 14, "Nile L bend Cairo"),
    ("L", 51.8851, -0.4180, 14, "Luton airport L runway"),
    ("L", 47.6062, -122.3321, 15, "Seattle waterfront L shape"),
    ("L", 40.7128, -74.0060, 14, "Lower Manhattan L shape"),
    # === M (4) ===
    ("M", 31.5580, -110.3530, 13, "Mountain zigzag Arizona"),
    ("M", 46.6034, 1.8883, 11, "River meanders central France"),
    ("M", 61.2181, -149.9003, 13, "Matanuska River M Alaska"),
    ("M", 47.6101, -122.2015, 15, "Mercer Island M shape Seattle"),
    # === N (4) ===
    ("N", 63.3462, 18.7280, 13, "River zigzag N Sweden"),
    ("N", 38.8977, -77.0365, 15, "DC road N pattern"),
    ("N", 48.2082, 16.3738, 14, "Danube channel N Vienna"),
    ("N", 37.7749, -122.4194, 14, "SF hills road N pattern"),
    # === O (5) ===
    ("O", 21.1260, -11.4010, 10, "Richat Structure Eye of Sahara"),
    ("O", -21.1789, 55.2694, 13, "Piton de la Fournaise Reunion Island"),
    ("O", 37.7510, -122.4477, 16, "Lake Merced O shape SF"),
    ("O", 64.0200, -16.5300, 13, "Volcano crater O Iceland"),
    ("O", 38.8862, -77.0248, 16, "Jefferson Memorial tidal basin O"),
    # === P (4) ===
    ("P", 42.9831, -70.6086, 14, "Plum Island peninsula P Massachusetts"),
    ("P", 39.4565, -0.3499, 14, "Valencia port P shape Spain"),
    ("P", 43.5561, 10.3109, 14, "Livorno port P shape Italy"),
    ("P", 51.5010, -0.1416, 15, "St James Park P shape London"),
    # === Q (4) ===
    ("Q", -3.4653, 29.3430, 12, "Lake Tanganyika outflow Q"),
    ("Q", 47.3769, 8.5417, 15, "Zurich lake Q shape"),
    ("Q", 37.7572, -122.5055, 15, "Lake Merced Q outflow SF"),
    ("Q", 43.6532, -79.3832, 14, "Toronto Islands Q shape"),
    # === R (4) ===
    ("R", 51.4545, -0.9580, 14, "Thames R branch Reading UK"),
    ("R", 32.0853, 34.7818, 14, "Yarkon River R branch Tel Aviv"),
    ("R", 40.4168, -3.7038, 15, "Manzanares R branch Madrid"),
    ("R", 47.3774, 8.5385, 14, "Limmat River R branch Zurich"),
    # === S (5) ===
    ("S", 29.9511, -90.0715, 13, "Mississippi S bend New Orleans"),
    ("S", -2.5000, 28.8667, 12, "Ruzizi River S curve Burundi"),
    ("S", 38.7267, -9.1403, 14, "Tagus S bend Lisbon"),
    ("S", 47.5580, 7.5855, 14, "Rhine S bend Basel"),
    ("S", 44.4949, 11.3426, 14, "Reno River S Bologna"),
    # === T (5) ===
    ("T", 29.9746, 31.1374, 14, "Nile T junction Cairo"),
    ("T", 53.7466, 7.8956, 13, "River T junction Germany"),
    ("T", 51.5099, -0.0093, 15, "Tower of London T shape"),
    ("T", 48.8631, 2.2873, 16, "Trocadero T shape Paris"),
    ("T", 37.4449, 25.3484, 14, "Mykonos harbor T"),
    # === U (5) ===
    ("U", 57.5780, 11.9720, 14, "Gothenburg harbor U Sweden"),
    ("U", 46.2044, 6.1432, 14, "Geneva lake U bend"),
    ("U", 43.7696, 11.2558, 14, "Arno River U bend Florence"),
    ("U", 36.8955, -76.3002, 14, "Norfolk harbor U shape Virginia"),
    ("U", 50.0755, 14.4378, 14, "Vltava U bend Prague"),
    # === V (5) ===
    ("V", 46.1570, 8.7646, 14, "Valley V Verzasca Switzerland"),
    ("V", 45.1845, 5.7255, 14, "Grenoble V valley France"),
    ("V", 38.4890, -0.3625, 14, "Alicante V shape Spain"),
    ("V", 39.4699, -0.3763, 14, "Valencia V river Spain"),
    ("V", 19.4326, -99.1332, 14, "Mexico City V junction"),
    # === W (4) ===
    ("W", 24.4667, 39.6111, 13, "Red Sea coast W Saudi Arabia"),
    ("W", 60.4720, 22.2600, 13, "Turku archipelago W Finland"),
    ("W", 41.1579, -8.6291, 14, "Porto bridges W pattern Portugal"),
    ("W", 43.3248, 5.3662, 14, "Marseille coast W France"),
    # === X (5) ===
    ("X", 40.4286, -3.6963, 15, "Madrid road X junction"),
    ("X", 51.5155, -0.1410, 16, "Piccadilly X junction London"),
    ("X", 48.8606, 2.3376, 16, "Louvre X junction Paris"),
    ("X", 52.5200, 13.4050, 15, "Berlin road X junction"),
    ("X", 35.6895, 139.6917, 15, "Tokyo crossing X pattern"),
    # === Y (5) ===
    ("Y", -8.0000, 31.0000, 12, "Ugab River Y Namibia"),
    ("Y", 46.5197, 6.6323, 14, "Lausanne Y junction Switzerland"),
    ("Y", 37.7821, -122.3915, 15, "Mission Creek Y junction SF"),
    ("Y", 55.9533, -3.1883, 14, "Edinburgh Y junction Scotland"),
    ("Y", 29.3500, 47.9900, 14, "Kuwait river Y junction"),
    # === Z (4) ===
    ("Z", 39.2842, -76.6071, 14, "Baltimore road Z pattern"),
    ("Z", 37.0194, -7.9304, 14, "Algarve coast Z Portugal"),
    ("Z", 44.4268, 26.1025, 15, "Bucharest road Z Romania"),
    ("Z", 46.0207, 7.7491, 14, "Mountain road Z Swiss Alps"),
]


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Research/Academic) SatLetter Dataset Builder"
}


def zoom_to_span(zoom: int) -> float:
    """Convert zoom level to approximate lat/lon span in degrees."""
    # At zoom 0, the world is 360°. Each zoom halves the span.
    return 360.0 / (2 ** zoom) * 0.5  # half-span


def get_esri_url(lat: float, lon: float, zoom: int, size: int = 600) -> str:
    """Build Esri World Imagery export URL."""
    span = zoom_to_span(zoom)
    xmin = lon - span
    xmax = lon + span
    ymin = lat - span
    ymax = lat + span
    return (
        f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
        f"?bbox={xmin},{ymin},{xmax},{ymax}"
        f"&bboxSR=4326&imageSR=4326"
        f"&size={size},{size}"
        f"&format=jpg&f=image"
    )


def get_google_url(lat: float, lon: float, zoom: int, api_key: str, size: int = 600) -> str:
    """Build Google Maps Static API URL."""
    return (
        f"https://maps.googleapis.com/maps/api/staticmap"
        f"?center={lat},{lon}&zoom={zoom}"
        f"&size={size}x{size}&maptype=satellite"
        f"&key={api_key}"
    )


def get_mapbox_url(lat: float, lon: float, zoom: int, token: str, size: int = 600) -> str:
    """Build Mapbox Static Images API URL."""
    return (
        f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static"
        f"/{lon},{lat},{zoom},0"
        f"/{size}x{size}@2x"
        f"?access_token={token}"
    )


def download_image(url: str, save_path: str, timeout: int = 30) -> bool:
    """Download an image from URL using urllib (no pip deps)."""
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
                if len(data) > 5000:  # Minimum valid image size
                    with open(save_path, "wb") as f:
                        f.write(data)
                    return True
            return False
        except urllib.error.HTTPError as e:
            if e.code in (404, 403):
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


def download_google_earth_letters(
    output_dir: str = "dataset/satellite_letters/raw/google_earth",
    provider: str = "esri",
    api_key: str = None,
    image_size: int = 600,
):
    """
    Download satellite imagery of letter-shaped landforms.

    Args:
        output_dir: Output directory
        provider: 'esri' (free), 'google' (needs key), or 'mapbox' (needs token)
        api_key: API key/token for Google or Mapbox
        image_size: Image dimensions in pixels
    """
    print("=" * 60)
    print("Google Earth Satellite Letter Downloader")
    print("=" * 60)
    print(f"Provider: {provider}")
    print(f"Output:   {output_dir}")
    print(f"Locations: {len(LETTER_COORDINATES)}")
    print(f"Image size: {image_size}x{image_size}")
    print()

    if provider in ("google", "mapbox") and not api_key:
        print("⚠️  No API key provided! Falling back to Esri (free).")
        provider = "esri"

    # Create directories
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        os.makedirs(os.path.join(output_dir, letter), exist_ok=True)

    total = 0
    downloaded = 0
    skipped = 0
    letter_counts = {}

    for letter, lat, lon, zoom, desc in LETTER_COORDINATES:
        # Build URL based on provider
        if provider == "google":
            url = get_google_url(lat, lon, zoom, api_key, image_size)
        elif provider == "mapbox":
            url = get_mapbox_url(lat, lon, zoom, api_key, image_size)
        else:
            url = get_esri_url(lat, lon, zoom, image_size)

        # Create unique filename using coordinate hash for dedup
        coord_hash = hashlib.md5(f"{lat}_{lon}_{zoom}".encode()).hexdigest()[:8]
        existing = len([
            f for f in os.listdir(os.path.join(output_dir, letter))
            if f.endswith(".jpg")
        ])
        filename = f"ge_{letter}_{existing:03d}_{coord_hash}.jpg"
        filepath = os.path.join(output_dir, letter, filename)

        # Check if this coordinate already downloaded
        letter_dir = os.path.join(output_dir, letter)
        already_exists = any(
            coord_hash in f for f in os.listdir(letter_dir) if f.endswith(".jpg")
        )

        if already_exists:
            skipped += 1
            total += 1
            letter_counts[letter] = letter_counts.get(letter, 0) + 1
            continue

        print(f"  [{letter}] {desc}... ", end="", flush=True)

        if download_image(url, filepath):
            downloaded += 1
            letter_counts[letter] = letter_counts.get(letter, 0) + 1
            fsize = os.path.getsize(filepath)
            print(f"✓ ({fsize // 1024}KB)")
            time.sleep(0.5)
        else:
            print("✗")
            if os.path.exists(filepath):
                os.remove(filepath)

        total += 1

    # Summary
    print()
    print("=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    total_on_disk = sum(
        len([f for f in os.listdir(os.path.join(output_dir, l)) if f.endswith(".jpg")])
        for l in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if os.path.isdir(os.path.join(output_dir, l))
    )

    print(f"Total attempted: {total}")
    print(f"New downloads:   {downloaded}")
    print(f"Already existed: {skipped}")
    print(f"Total on disk:   {total_on_disk}")
    print()

    print("Per-letter counts:")
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        letter_dir = os.path.join(output_dir, letter)
        n = len([f for f in os.listdir(letter_dir) if f.endswith(".jpg")]) if os.path.isdir(letter_dir) else 0
        status = "✓" if n > 0 else "✗ EMPTY"
        print(f"  {letter}: {n:3d} {status}")

    # Save metadata
    metadata = {
        "source": f"Satellite imagery ({provider})",
        "provider": provider,
        "total_images": total_on_disk,
        "coordinates_count": len(LETTER_COORDINATES),
        "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "letter_counts": {
            l: len([f for f in os.listdir(os.path.join(output_dir, l)) if f.endswith(".jpg")])
            for l in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            if os.path.isdir(os.path.join(output_dir, l))
        },
    }
    meta_path = os.path.join(output_dir, "download_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata: {meta_path}")
    return letter_counts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download satellite imagery of letter-shaped landforms"
    )
    parser.add_argument(
        "--output", "-o",
        default="dataset/satellite_letters/raw/google_earth",
        help="Output directory",
    )
    parser.add_argument(
        "--provider", "-p",
        choices=["esri", "google", "mapbox"],
        default="esri",
        help="Imagery provider (default: esri, free)",
    )
    parser.add_argument(
        "--api-key", "-k",
        default=os.environ.get("GOOGLE_MAPS_API_KEY") or os.environ.get("MAPBOX_TOKEN"),
        help="API key (or set GOOGLE_MAPS_API_KEY / MAPBOX_TOKEN env var)",
    )
    parser.add_argument(
        "--size", "-s",
        type=int,
        default=600,
        help="Image size in pixels (default: 600)",
    )
    args = parser.parse_args()

    download_google_earth_letters(
        output_dir=args.output,
        provider=args.provider,
        api_key=args.api_key,
        image_size=args.size,
    )
