#!/usr/bin/env bash
# Convert oddadmix's 7M Kokoro distills into sherpa-onnx bundles.
#
# Produces, per model:
#   model.fp32.onnx   ~30 MB, iSTFT notch baked in, metadata stamped
#   model.int8.onnx   ~8.7 MB
#   tokens.txt        mirrors the TRAINED vocab (config.json), not the
#                     official 178-token table
#   voices.bin        af_msa style pack, flat float32 [510, 1, 256]
set -ex

python3 -m pip install -q onnx onnxruntime torch huggingface_hub soundfile misaki

# ----------------------------------------------------------------- Arabic
huggingface-cli download oddadmix/Nabra-7M-Distill --local-dir ./nabra_7m

mkdir -p nabra-7m-distill
python3 export_onnx.py \
  --repo oddadmix/Nabra-7M-Distill --dir nabra_7m \
  --weights kokoro_arabic_7m.pth --out nabra-7m-distill/model.fp32.onnx

python3 add_meta_data.py \
  --model nabra-7m-distill/model.fp32.onnx --lang ar \
  --language "Arabic (Modern Standard)" \
  --comment "Nabra-7M-Distill Arabic FP32 with baked-in iSTFT notch FIR"

python3 dynamic_quantization.py \
  --src nabra-7m-distill/model.fp32.onnx \
  --dst nabra-7m-distill/model.int8.onnx

python3 generate_assets.py --lang ar --dir nabra_7m --out nabra-7m-distill

# ---------------------------------------------------------------- English
huggingface-cli download oddadmix/Kokoro-7M-Distill --local-dir ./kokoro_7m

mkdir -p kokoro-7m-distill
python3 export_onnx.py \
  --repo oddadmix/Kokoro-7M-Distill --dir kokoro_7m \
  --weights kokoro_en_7m.pth --out kokoro-7m-distill/model.fp32.onnx

python3 add_meta_data.py \
  --model kokoro-7m-distill/model.fp32.onnx --lang en-us \
  --language "English" \
  --comment "Kokoro-7M-Distill English FP32 with baked-in iSTFT notch FIR"

python3 dynamic_quantization.py \
  --src kokoro-7m-distill/model.fp32.onnx \
  --dst kokoro-7m-distill/model.int8.onnx

python3 generate_assets.py --lang en --dir kokoro_7m --out kokoro-7m-distill

# The English model needs misaki phonemes; sherpa's espeak front-end does not
# match it (see README). Ship the official lexicon so the bundle is ready the
# moment the runtime can consult it.
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2
tar xf kokoro-multi-lang-v1_0.tar.bz2 kokoro-multi-lang-v1_0/lexicon-us-en.txt
cp kokoro-multi-lang-v1_0/lexicon-us-en.txt kokoro-7m-distill/

ls -lh nabra-7m-distill kokoro-7m-distill
