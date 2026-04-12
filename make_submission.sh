#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$ROOT_DIR/submission_pkg"
ZIP_PATH="$ROOT_DIR/submission.zip"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
cp "$ROOT_DIR/agent.py" "$OUT_DIR/agent.py"
cp "$ROOT_DIR/weights.pth" "$OUT_DIR/weights.pth"
rm -f "$ZIP_PATH"
(
  cd "$OUT_DIR"
  zip -r "$ZIP_PATH" agent.py weights.pth >/dev/null
)
echo "Created: $ZIP_PATH"
echo "Contents:"
unzip -l "$ZIP_PATH"
