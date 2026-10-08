#!/usr/bin/env bash
# Download the public datasets that can be fetched without an account.
# Everything goes under ~/datasets (change with DATASETS=/path ./fetch_datasets.sh).
set -e
DATASETS="${DATASETS:-$HOME/datasets}"
mkdir -p "$DATASETS"
cd "$DATASETS"

# 1. PlantVillage: 54k lab photos, 38 crop/disease classes
if [ ! -d PlantVillage-Dataset/raw/color ]; then
  echo "== PlantVillage"
  git clone --depth 1 --filter=blob:none --sparse https://github.com/spMohanty/PlantVillage-Dataset.git
  (cd PlantVillage-Dataset && git sparse-checkout set raw/color)
fi

# 2. PlantDoc: ~2,500 real field photos, 27 classes (fixes the lab-photo problem)
if [ ! -d PlantDoc-Dataset ]; then
  echo "== PlantDoc"
  git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Dataset.git
fi

# 3. Imagenette: ~13k everyday photos, used as "not relevant"
if [ ! -d imagenette2-320 ]; then
  echo "== Imagenette"
  curl -fL https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz -o imagenette2-320.tgz
  tar xzf imagenette2-320.tgz && rm imagenette2-320.tgz
fi

echo
echo "Done. Datasets in $DATASETS:"
du -sh "$DATASETS"/* 2>/dev/null
echo "Check each dataset's license file before sharing a model trained on it."
