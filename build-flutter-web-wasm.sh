#!/usr/bin/env bash
# Copyright (c)  2026  Xiaomi Corporation
#
# Build the WASM module for Flutter web and copy assets to the web plugin.
# JS wrapper files are symlinked (not copied) to avoid code duplication.
# For publishing, see release-dart-package.yaml which replaces symlinks
# with real files.

set -ex

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

# Build the WASM module
./build-wasm-simd-web.sh

# Copy WASM module to the Flutter web plugin's assets/
cp build-wasm-simd-web/install/bin/wasm/web/sherpa-onnx-wasm-web.js \
   flutter/sherpa_onnx_web/assets/

cp build-wasm-simd-web/install/bin/wasm/web/sherpa-onnx-wasm-web.wasm \
   flutter/sherpa_onnx_web/assets/

# Create symlinks for JS wrappers (avoid code duplication)
cd flutter/sherpa_onnx_web/assets
ln -sf ../../../wasm/asr/sherpa-onnx-asr.js .
ln -sf ../../../wasm/tts/sherpa-onnx-tts.js .
ln -sf ../../../wasm/vad/sherpa-onnx-vad.js .
ln -sf ../../../wasm/kws/sherpa-onnx-kws.js .
ln -sf ../../../wasm/nodejs/sherpa-onnx-punctuation.js .
ln -sf ../../../wasm/speaker-diarization/sherpa-onnx-speaker-diarization.js .
ln -sf ../../../wasm/speech-enhancement/sherpa-onnx-speech-enhancement.js .
cd "$SCRIPT_DIR"

echo ""
echo "Done! WASM assets copied and JS symlinks created."
echo "You can now run: cd flutter-examples/hello_world && flutter run -d chrome"
