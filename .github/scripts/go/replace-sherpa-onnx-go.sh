#!/usr/bin/env bash
# Replace the published sherpa_onnx.go in the Go module cache with the local
# version.  Must be called AFTER "go mod tidy" (which downloads the module)
# and BEFORE "go build".
#
# On Windows x86 the published file contains a [1073741824]float32 array that
# exceeds the 32-bit address space; the local version uses unsafe.Slice instead.
set -e

gopath=$(go env GOPATH)
local_file="$(cd "$(dirname "$0")/../.." && pwd)/scripts/go/sherpa_onnx.go"

if [ ! -f "$local_file" ]; then
  echo "[replace-sherpa-onnx-go] local file not found: $local_file — skipping"
  exit 0
fi

found=0
for f in $(find "$gopath/pkg/mod/github.com/k2-fsa/" \
              -name "sherpa_onnx.go" -path "*/sherpa-onnx-go*" 2>/dev/null); do
  echo "[replace-sherpa-onnx-go] Replacing $f"
  chmod u+w "$f"
  cp -v "$local_file" "$f"
  found=1
done

if [ "$found" -eq 0 ]; then
  echo "[replace-sherpa-onnx-go] No published sherpa_onnx.go found in module cache — skipping"
fi
