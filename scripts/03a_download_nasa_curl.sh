#!/bin/bash
# ============================================================
# Script 03a: Download NASA Landsat Alphabet images (curl)
# ============================================================
# Alternative to 03_scrape_nasa_landsat.py for environments
# where Python requests doesn't have internet access.
#
# URL pattern (verified):
#   https://science.nasa.gov/specials/your-name-in-landsat/images/{letter}_{index}.jpg
#
# Verified image counts per letter:
#   a:5 b:2 c:3 d:2 e:4 f:2 g:1 h:2 i:5 j:3 k:2 l:4 m:3
#   n:3 o:2 p:2 q:2 r:4 s:3 t:2 u:2 v:4 w:2 x:3 y:3 z:2
#   Total: 71 images
# ============================================================

BASE_URL="https://science.nasa.gov/specials/your-name-in-landsat/images"
OUTPUT_DIR="${1:-dataset/satellite_letters/raw/nasa}"

# Letter counts (verified via JS HEAD requests)
declare -A COUNTS=(
  [a]=5 [b]=2 [c]=3 [d]=2 [e]=4 [f]=2 [g]=1 [h]=2 [i]=5
  [j]=3 [k]=2 [l]=4 [m]=3 [n]=3 [o]=2 [p]=2 [q]=2 [r]=4
  [s]=3 [t]=2 [u]=2 [v]=4 [w]=2 [x]=3 [y]=3 [z]=2
)

TOTAL=0
DOWNLOADED=0
SKIPPED=0

echo "============================================================"
echo "NASA Landsat Alphabet Downloader (curl)"
echo "============================================================"
echo "Source: ${BASE_URL}"
echo "Output: ${OUTPUT_DIR}"
echo ""

for letter in {a..z}; do
  UPPER=$(echo "$letter" | tr '[:lower:]' '[:upper:]')
  LETTER_DIR="${OUTPUT_DIR}/${UPPER}"
  mkdir -p "$LETTER_DIR"

  count=${COUNTS[$letter]}
  echo -n "[${UPPER}] Downloading ${count} images... "

  letter_dl=0
  for ((i=0; i<count; i++)); do
    idx=$(printf "%03d" $i)
    filename="nasa_landsat_${UPPER}_${idx}.jpg"
    filepath="${LETTER_DIR}/${filename}"
    url="${BASE_URL}/${letter}_${i}.jpg"

    if [ -f "$filepath" ] && [ -s "$filepath" ]; then
      letter_dl=$((letter_dl + 1))
      SKIPPED=$((SKIPPED + 1))
    else
      if curl -sf -o "$filepath" \
        -H "User-Agent: Mozilla/5.0 (Research/Academic) SatLetter" \
        "$url" 2>/dev/null; then
        letter_dl=$((letter_dl + 1))
        DOWNLOADED=$((DOWNLOADED + 1))
        sleep 0.3
      else
        echo -n "✗ "
        rm -f "$filepath"  # cleanup empty files
      fi
    fi
    TOTAL=$((TOTAL + 1))
  done

  echo "${letter_dl}/${count} ✓"
done

echo ""
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
echo "Total attempted: ${TOTAL}"
echo "New downloads:   ${DOWNLOADED}"
echo "Already existed: ${SKIPPED}"
echo "Total on disk:   $(find "${OUTPUT_DIR}" -name '*.jpg' -type f | wc -l)"
echo ""
echo "Per-letter breakdown:"
for letter in {a..z}; do
  UPPER=$(echo "$letter" | tr '[:lower:]' '[:upper:]')
  n=$(find "${OUTPUT_DIR}/${UPPER}" -name '*.jpg' -type f 2>/dev/null | wc -l)
  echo "  ${UPPER}: ${n} images"
done
