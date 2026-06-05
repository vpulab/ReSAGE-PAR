#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="$ROOT_DIR/third_party"
mkdir -p "$TARGET_DIR"

TMPDIR="$(mktemp -d)"
cleanup() { rm -rf "$TMPDIR"; }
trap cleanup EXIT

echo "Fetching files into $TARGET_DIR"

echo "Cloning sayakpaul/cmmd-pytorch (shallow)..."
git clone --depth 1 https://github.com/sayakpaul/cmmd-pytorch.git "$TMPDIR/cmmd" >/dev/null 2>&1 || true

FILES=("io_util.py" "distance.py" "embedding.py")
SRC_CANDIDATES=("metrics" "src" "cmmd" "cmmd_pytorch" "")
for f in "${FILES[@]}"; do
  found=false
  for d in "${SRC_CANDIDATES[@]}"; do
    srcpath="$TMPDIR/cmmd/$d/$f"
    if [ -f "$srcpath" ]; then
      cp "$srcpath" "$TARGET_DIR/$f"
      echo "Copied $f from $d"
      found=true
      break
    fi
  done

  if [ "$found" = false ]; then
    # try raw URLs on common branches/paths
    for br in main master; do
      url1="https://raw.githubusercontent.com/sayakpaul/cmmd-pytorch/$br/metrics/$f"
      url2="https://raw.githubusercontent.com/sayakpaul/cmmd-pytorch/$br/$f"
      if curl -fsSL "$url1" -o "$TARGET_DIR/$f"; then
        echo "Downloaded $f from $url1"
        found=true
        break
      fi
      if curl -fsSL "$url2" -o "$TARGET_DIR/$f"; then
        echo "Downloaded $f from $url2"
        found=true
        break
      fi
    done
  fi

  if [ "$found" = false ]; then
    echo "Warning: could not locate $f in sayakpaul/cmmd-pytorch"
  fi
done


echo "Cloning justin4ai/FD-DINOv2 into third_party/FD-DINOv2 (shallow)..."
if git clone --depth 1 https://github.com/justin4ai/FD-DINOv2.git "$TARGET_DIR/FD-DINOv2" >/dev/null 2>&1; then
  echo "Cloned FD-DINOv2 into $TARGET_DIR/FD-DINOv2"
else
  echo "Warning: could not clone FD-DINOv2 into $TARGET_DIR/FD-DINOv2"
fi

echo "Done. Files present in $TARGET_DIR:"
ls -la "$TARGET_DIR"

exit 0
