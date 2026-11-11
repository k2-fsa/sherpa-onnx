# Introduction

Please refer to
https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
to download a model.

## Text frontend

Choose exactly one of the two text frontends below (not both):

- **option 1 (default): phonemize** — the text is phonemized with
  piper-phonemize; no extra *model* files are needed. Download the wasm bundle
  and unzip it so the files end up in `wasm/tts/piper-phonemize/`:

  ```bash
  cd sherpa-onnx/wasm/tts
  wget -q https://github.com/csukuangfj/piper-phonemize/releases/download/wasm/piper-phonemize-v1.5.9-wasm-browser.zip
  unzip -q piper-phonemize-v1.5.9-wasm-browser.zip
  mv browser piper-phonemize
  rm piper-phonemize-v1.5.9-wasm-browser.zip
  ```

- **option 2: lexicon** — set `usePhonemize = false` in
  `../sherpa-onnx-tts.js` and put a lexicon next to the model files:

  ```bash
  cd sherpa-onnx/wasm/tts/assets
  wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/lexicon-en-us.txt
  ```

Neither option uses the espeak-ng-data directory shipped in the model
package; you can leave it out to keep the download small.

Some models bring their own text frontend and need neither option above:
PocketTTS (`modelType` 5) and SupertonicTTS (`modelType` 7). For those, skip
this whole section.

Please set `let modelType = 0` to the number matching your model in
`../sherpa-onnx-tts.js` (0 vits, 1/2/3 matcha, 4 zipvoice, 5 pocket,
6 kitten, 7 supertonic, 8 kokoro-multi-lang). The demo UI adapts itself
to the model type.

## Model files

The following is an example:
```bash
cd sherpa-onnx/wasm/tts/assets

wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-libritts_r-medium.tar.bz2
tar xf vits-piper-en_US-libritts_r-medium.tar.bz2
rm vits-piper-en_US-libritts_r-medium.tar.bz2
mv vits-piper-en_US-libritts_r-medium/en_US-libritts_r-medium.onnx ./model.onnx
mv vits-piper-en_US-libritts_r-medium/tokens.txt ./
rm -rf vits-piper-en_US-libritts_r-medium
```

ZipVoice example:

```bash
cd sherpa-onnx/wasm/tts/assets

wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2
tar xf sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2
rm sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2

mv sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/encoder.int8.onnx ./
mv sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/decoder.int8.onnx ./
mv sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/tokens.txt ./
mv sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/lexicon.txt ./
rm -rf sherpa-onnx-zipvoice-distill-int8-zh-en-emilia

wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos_24khz.onnx
```

PocketTTS example:

```bash
cd sherpa-onnx/wasm/tts/assets

wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
tar xf sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
rm sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2

mv sherpa-onnx-pocket-tts-int8-2026-01-26/lm_flow.int8.onnx ./
mv sherpa-onnx-pocket-tts-int8-2026-01-26/lm_main.int8.onnx ./
mv sherpa-onnx-pocket-tts-int8-2026-01-26/encoder.onnx ./
mv sherpa-onnx-pocket-tts-int8-2026-01-26/decoder.int8.onnx ./
mv sherpa-onnx-pocket-tts-int8-2026-01-26/text_conditioner.onnx ./
mv sherpa-onnx-pocket-tts-int8-2026-01-26/vocab.json ./
mv sherpa-onnx-pocket-tts-int8-2026-01-26/token_scores.json ./
rm -rf sherpa-onnx-pocket-tts-int8-2026-01-26
```

SupertonicTTS example (31 languages; own text frontend, so no lexicon and no
piper-phonemize are needed):

```bash
cd sherpa-onnx/wasm/tts/assets

wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2
tar xf sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2
rm sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2

mv sherpa-onnx-supertonic-3-tts-int8-2026-05-11/duration_predictor.int8.onnx ./
mv sherpa-onnx-supertonic-3-tts-int8-2026-05-11/text_encoder.int8.onnx ./
mv sherpa-onnx-supertonic-3-tts-int8-2026-05-11/vector_estimator.int8.onnx ./
mv sherpa-onnx-supertonic-3-tts-int8-2026-05-11/vocoder.int8.onnx ./
mv sherpa-onnx-supertonic-3-tts-int8-2026-05-11/tts.json ./
mv sherpa-onnx-supertonic-3-tts-int8-2026-05-11/unicode_indexer.bin ./
mv sherpa-onnx-supertonic-3-tts-int8-2026-05-11/voice.bin ./
rm -rf sherpa-onnx-supertonic-3-tts-int8-2026-05-11

# in ../sherpa-onnx-tts.js, set: let modelType = 7
```

You should have the following files in `assets` before you can run
`build-wasm-simd-tts.sh`

```
assets fangjun$ tree -L 1
.
├── README.md
├── model.onnx
└── tokens.txt

1 directory, 2 files
```

You can find example build scripts at:

  - English TTS: https://github.com/k2-fsa/sherpa-onnx/blob/master/.github/workflows/wasm-simd-hf-space-en-tts.yaml
  - German TTS: https://github.com/k2-fsa/sherpa-onnx/blob/master/.github/workflows/wasm-simd-hf-space-de-tts.yaml
