#!/usr/bin/env bash
#
# setup-ios.sh — Download the sherpa-onnx xcframework for iOS Tauri builds.
#
# Usage:
#   ./setup-ios.sh          # run once before `cargo tauri ios init`
#
# This script downloads the pre-built sherpa-onnx shared xcframework from
# GitHub Releases and places it in src-tauri/ so that Xcode can find it
# via `bundle.ios.frameworks` in tauri.conf.json.
#
# The xcframework is cached — subsequent runs are no-ops if it already exists.
#
# Why is this needed?
#   Xcode checks for xcframework existence before running any build phases.
#   `build.rs` downloads the xcframework during the build (too late for Xcode).
#   This script bridges the gap by downloading it before the first build.
#
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Auto-detect version from CMakeLists.txt (search upward from SCRIPT_DIR).
# Skip if VERSION is already set in the environment.
if [ -z "${VERSION:-}" ]; then
  dir="$SCRIPT_DIR"
  while [ "$dir" != "/" ]; do
    if [ -f "$dir/CMakeLists.txt" ]; then
      VERSION=$(grep 'SHERPA_ONNX_VERSION' "$dir/CMakeLists.txt" | head -1 | sed 's/.*"\(.*\)".*/\1/')
      if [ -n "$VERSION" ]; then
        break
      fi
    fi
    dir="$(dirname "$dir")"
  done
fi

if [ -z "$VERSION" ]; then
  echo "Error: Cannot determine sherpa-onnx version."
  echo "Searched upward from: $SCRIPT_DIR"
  echo "Please set VERSION environment variable manually, e.g.:"
  echo "  VERSION=1.13.6 ./setup-ios.sh"
  exit 1
fi

DEST="$SCRIPT_DIR"
XCFRAMEWORK="$DEST/sherpa-onnx.xcframework"

if [ -d "$XCFRAMEWORK" ]; then
  echo "sherpa-onnx.xcframework already exists at $XCFRAMEWORK"
  exit 0
fi

URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/xcframework/sherpa-onnx-v${VERSION}-ios-shared-onnxruntime-static.xcframework.zip"
echo "Downloading sherpa-onnx xcframework v${VERSION}..."
echo "  $URL"

TMPFILE=$(mktemp /tmp/sherpa-onnx-XXXXXX.zip)
trap 'rm -f "$TMPFILE"' EXIT

curl --fail -L -o "$TMPFILE" "$URL"
unzip -q "$TMPFILE" -d "$DEST"

echo "Installed sherpa-onnx.xcframework to $XCFRAMEWORK"
