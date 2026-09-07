#!/usr/bin/env bash
# On Windows, copy the published DLLs into scripts/go/_internal/lib/ and add a
# go.mod replace directive so that the sherpa_onnx package is resolved from the
# local _internal directory (which has the fixed sherpa_onnx.go using
# unsafe.Slice instead of a [1073741824]float32 fixed-size array).
#
# Must be called AFTER "go mod tidy" (which downloads the module)
# and BEFORE "go build".
set -e

goos=$(go env GOOS)
if [[ "$goos" != "windows" ]]; then
  echo "[replace-sherpa-onnx-go] Not Windows ($goos), skipping"
  exit 0
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
internal_dir="$repo_root/scripts/go/_internal"

if [ ! -f "$internal_dir/sherpa_onnx.go" ]; then
  echo "[replace-sherpa-onnx-go] $internal_dir/sherpa_onnx.go not found — skipping"
  exit 0
fi

gopath=$(go env GOPATH)
goarch=$(go env GOARCH)

if [[ "$goarch" == "386" ]]; then
  win_lib_dir=i686-pc-windows-gnu
else
  win_lib_dir=x86_64-pc-windows-gnu
fi

# Find the published sherpa-onnx-go-windows module in the cache
windows_mod=$(find "$gopath/pkg/mod/github.com/k2-fsa/" \
  -maxdepth 1 -name "sherpa-onnx-go-windows@*" -type d 2>/dev/null | head -1)

if [ -z "$windows_mod" ]; then
  echo "[replace-sherpa-onnx-go] No published sherpa-onnx-go-windows found — skipping"
  exit 0
fi

# Copy DLLs from the published module into _internal/lib/<arch>/
mkdir -p "$internal_dir/lib/$win_lib_dir"
cp -v "$windows_mod/lib/$win_lib_dir"/* "$internal_dir/lib/$win_lib_dir/" 2>/dev/null || true

# Point go.mod at the local _internal directory (same approach as test-go.yaml).
# The _internal directory already contains the fixed sherpa_onnx.go and the
# platform-specific build_*.go files with the correct #cgo LDFLAGS.
go mod edit -replace "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx=$internal_dir"

echo "[replace-sherpa-onnx-go] Done. Using $internal_dir"
