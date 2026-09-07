#!/usr/bin/env bash
# Replace the published sherpa_onnx.go with a local version that uses
# unsafe.Slice instead of fixed-size arrays.
#
# On Windows the published file contains a [1073741824]float32 array (4 GB)
# that causes compilation failures on x86 and runtime allocation failures
# on x64.  The local version uses unsafe.Slice instead.
#
# Must be called AFTER "go mod tidy" (which downloads the module)
# and BEFORE "go build".
set -e

local_file="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/scripts/go/sherpa_onnx.go"

if [ ! -f "$local_file" ]; then
  echo "[replace-sherpa-onnx-go] local file not found: $local_file — skipping"
  exit 0
fi

goos=$(go env GOOS)
if [[ "$goos" != "windows" ]]; then
  echo "[replace-sherpa-onnx-go] Not Windows ($goos), skipping"
  exit 0
fi

gopath=$(go env GOPATH)

# Find the published sherpa-onnx-go-windows module in the cache
windows_mod=$(find "$gopath/pkg/mod/github.com/k2-fsa/" \
  -maxdepth 1 -name "sherpa-onnx-go-windows@*" -type d 2>/dev/null | head -1)

if [ -z "$windows_mod" ]; then
  echo "[replace-sherpa-onnx-go] No published sherpa-onnx-go-windows found — skipping"
  exit 0
fi

# Create a local copy of the module with the fixed sherpa_onnx.go
local_dir="./_sherpa_onnx_go_windows"
rm -rf "$local_dir"
cp -r "$windows_mod" "$local_dir"
chmod -R u+w "$local_dir"
cp -v "$local_file" "$local_dir/sherpa_onnx.go"

# The local sherpa_onnx.go has no #cgo LDFLAGS — those live in separate
# build_*.go files under scripts/go/_internal/.  Copy them so the linker
# can find the DLLs at build time.
internal_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/scripts/go/_internal"
cp -v "$internal_dir"/build_*.go "$local_dir/" 2>/dev/null || true

# Use go mod edit -replace so go build uses the local copy.
# This is more reliable than modifying the module cache, especially on Windows.
go mod edit -replace "github.com/k2-fsa/sherpa-onnx-go-windows=./_sherpa_onnx_go_windows"

echo "[replace-sherpa-onnx-go] Done. Using local copy at $local_dir"
