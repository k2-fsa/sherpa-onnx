#!/usr/bin/env bash
set -euo pipefail

echo "=== Building sherpa-onnx ==="
cargo build -p sherpa-onnx

echo "=== Checking code with cargo check ==="
cargo check -p sherpa-onnx

echo "=== Running clippy for lints ==="
cargo clippy -p sherpa-onnx -- -D warnings

echo "=== Running tests ==="
cargo test -p sherpa-onnx

echo "All checks passed for sherpa-onnx ✅"
