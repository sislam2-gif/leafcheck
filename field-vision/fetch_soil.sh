#!/usr/bin/env bash
# Download public soil photo datasets into ~/datasets (change with DATASETS=/path).
#   ./fetch_soil.sh            soil texture (Loam / Sand / Clay), 1.7 GB
#   ./fetch_soil.sh --types    also 7 soil types (alluvial, black, laterite, ...), non-commercial license
set -e
DATASETS="${DATASETS:-$HOME/datasets}"
mkdir -p "$DATASETS" && cd "$DATASETS"

# 1. Soil texture: 92k field photo patches, CC BY 4.0
#    https://zenodo.org/records/20404766
if [ ! -d soil-texture-92k ]; then
  echo "== Soil texture (Loam / Sand / Clay), 1.7 GB"
  curl -fL --retry 3 -C - "https://zenodo.org/records/20404766/files/dataset_v3_patches.zip?download=1" \
       -o soil-texture-92k.zip
  echo "4f64a3ed48015d37e5cd6946f9d2b61b  soil-texture-92k.zip" | md5sum -c -
  mkdir -p soil-texture-92k && unzip -q soil-texture-92k.zip -d soil-texture-92k && rm soil-texture-92k.zip
fi
echo "Folders in soil-texture-92k:"
find soil-texture-92k -maxdepth 3 -type d | head -20

# 2. Soil types (optional): 7 classes, CC BY-NC 4.0
if [ "$1" = "--types" ] && [ ! -d Soil-Classification-Dataset ]; then
  echo "== Soil types (7 classes)"
  git clone --depth 1 https://github.com/Phantom-fs/Soil-Classification-Dataset.git
fi

echo
echo "Done. Next, see 'Soil models' in README.md"
