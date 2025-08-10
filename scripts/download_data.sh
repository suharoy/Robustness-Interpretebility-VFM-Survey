#!/usr/bin/env bash
# Download datasets for the VFM Robustness project using the Kaggle CLI.
# Requires:
#   pip install kaggle
#   Place kaggle.json at ~/.kaggle/kaggle.json  (Windows Git Bash: /c/Users/<you>/.kaggle/kaggle.json)

set -euo pipefail

# ----- helpers -----
need() { command -v "$1" >/dev/null 2>&1 || { echo "Error: '$1' is required."; exit 1; }; }
unzip_one() {
  local zipfile="$1" out="$2"
  if command -v unzip >/dev/null 2>&1; then
    unzip -q -o "$zipfile" -d "$out"
  else
    # Fallback unzip via Python (if 'unzip' is not available)
    python3 - "$zipfile" "$out" <<'PY'
import sys, zipfile, os
zf, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
with zipfile.ZipFile(zf) as z:
    z.extractall(out)
PY
  fi
}

download_and_unpack() {
  local slug="$1" out="$2"
  mkdir -p "$out"
  echo "[info] Downloading $slug -> $out"
  kaggle datasets download -d "$slug" -p "$out" -q
  shopt -s nullglob
  for z in "$out"/*.zip; do
    echo "[info] Unzipping $(basename "$z")"
    unzip_one "$z" "$out"
    rm -f "$z"
  done
  echo "[done] $slug"
}

# ----- checks -----
need kaggle
# python3 is only needed if 'unzip' is missing; we don't fail here.

# ----- datasets -----
# CIFAR-10
download_and_unpack "petitbonney/cifar10-image-recognition" "data/cifar10"

# Fashion-MNIST
download_and_unpack "zalando-research/fashionmnist" "data/fashionmnist"

# ImageNet-100 (Kaggle copy)
download_and_unpack "ambityga/imagenet100" "data/imagenet100"

# Tiny-ImageNet
download_and_unpack "akash2sharma/tiny-imagenet" "data/tiny-imagenet"

# ImageNet-V2 (test set)
download_and_unpack "derrickdaniel/imagenet-v2-test" "data/imagenetv2"

echo
echo "All done. Tree under ./data:"
# show one level for sanity
find data -maxdepth 2 -type d -print | sed 's/^\.\///'