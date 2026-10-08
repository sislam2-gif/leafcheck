#!/usr/bin/env bash
# Download ONNX Runtime Web (the in-browser model runner) into a folder,
# so the model works with no internet. Usage: ./get_ort.sh exports/v1/vendor
set -e
DEST="${1:-vendor}"
VER="${ORT_VERSION:-1.22.0}"
BASE="https://cdn.jsdelivr.net/npm/onnxruntime-web@${VER}/dist"
mkdir -p "$DEST"
for f in ort.wasm.min.js ort-wasm-simd-threaded.wasm ort-wasm-simd-threaded.mjs; do
  echo "Downloading $f"
  curl -fsSL "$BASE/$f" -o "$DEST/$f"
done
echo "Done: $(du -sh "$DEST" | cut -f1) in $DEST"
