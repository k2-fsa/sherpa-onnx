# Forced-alignment Web App (WASM)

A pure-browser forced-alignment app: pick an audio/video file (or record from
the microphone), enter its transcript, and get the start/end time of every
word. Everything runs locally in the browser via WebAssembly — no backend,
no uploads. The UI is a single card-based page with a model-loading
progress bar.

Pre-built apps (one per model) are in the
[react-fa](https://github.com/k2-fsa/sherpa-onnx/releases/tag/react-fa)
release; [scripts/mfa/web.md](../../scripts/mfa/web.md) describes which one
to use.

## What the app does

- **Example** — loads the model package's test wav and fills in its
  transcript; try the whole flow without any external data.
- **Record** — record from the microphone and align the recording.
- **Select file** — drag & drop or pick an audio/video file (wav / mp3 / m4a /
  flac / ogg / mp4 / …; anything the browser's `decodeAudioData` handles).
- **Align** — runs the forced aligner (sherpa-onnx, WASM, CPU) and shows a
  word table (Word | Time | Audio) — click a row to play just that word (cut
  sample-accurately from the decoded audio, never including the neighbors),
  save a word's wav, or search the words. Views: Table / Timestamped /
  Plain.
- **Preview** — whole-file playback with the current word shown as a subtitle.
- **Export / Share** — `.TextGrid`, `_timestamped.txt`, `.txt`,
  `.TextGrid.txt` (WeChat-friendly), `.words.txt` (`word start -- end` per
  line).
- **Stats** — audio duration · decode · align · RTF.

## How it works

- The wasm module is `wasm/forced-alignment/` (built by
  `./build-wasm-simd-forced-alignment.sh` at the repo root): an emscripten
  package of the sherpa-onnx C API with the model files packed into
  `sherpa-onnx-wasm-main-fa.data` (see `wasm/forced-alignment/assets/README.md`).
- `src/lib/sherpa/engine.ts` loads the package in 4 phases (wrapper script →
  `.wasm` → `.data` → create the `ForcedAligner`), with a progress bar.
- The model is chosen by the build-time constant `SELECTED_MODEL_INDEX` in
  `src/lib/model-config.ts` (same order as the Android app's `ModelConfig.kt`).
  **It is not exposed to users** — CI builds one app per model.

## Build and run (development)

Prerequisites: Node.js `^20.19.0` or `>= 22.12.0` (what Vite requires), and
[emsdk](https://github.com/emscripten-core/emsdk)
**4.0.23** for the wasm module.

```bash
# 1. build the wasm module (once per model you want to bundle)
cd <repo-root>
export EMSCRIPTEN=/path/to/emsdk/upstream/emscripten   # or: source /path/to/emsdk/emsdk_env.sh
cd wasm/forced-alignment/assets
wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/zh-aishell3-pinyin.tar.bz2
tar xf zh-aishell3-pinyin.tar.bz2 && rm zh-aishell3-pinyin.tar.bz2
cd ../../..
./build-wasm-simd-forced-alignment.sh

# 2. copy the wasm package into the app
mkdir -p react-examples/forced-alignment/public/sherpa
cp build-wasm-simd-forced-alignment/install/bin/wasm/forced-alignment/sherpa-onnx-wasm-main-fa.{js,wasm,data} \
   react-examples/forced-alignment/public/sherpa/
cp wasm/nodejs/sherpa-onnx-forced-aligner.js react-examples/forced-alignment/public/sherpa/

# 3. run the dev server
cd react-examples/forced-alignment
npm install
npm run dev        # http://localhost:5173
```

Put several model directories into `wasm/forced-alignment/assets/` before
building the wasm to bundle several models; `SELECTED_MODEL_INDEX` picks one.

## Build the static bundle

```bash
cd react-examples/forced-alignment
./build-static.sh <model_index>       # 0..5, see the table in scripts/mfa/web.md
```

This produces a self-contained directory (default `dist-model-<index>/`):

```
dist-model-<index>/
├── index.html
├── assets/…                       # app js + css
└── sherpa/…                       # sherpa-onnx wasm + js (model inside .data)
```

Host it with **any static HTTP server** (GitHub Pages, nginx, S3,
`python3 -m http.server`, ...). Requirements:

- serve `.wasm` as `application/wasm` (most servers do; for nginx add
  `types { application/wasm wasm; }`);
- no backend, no special headers (the wasm is single-threaded SIMD; no
  COOP/COEP needed);
- total size is roughly the model package (~3–37 MB) + ~300 KB of app code.

## CI releases

`.github/workflows/react-fa.yaml` builds the 6 model variants and releases
them to the [react-fa](https://github.com/k2-fsa/sherpa-onnx/releases/tag/react-fa)
tag as `sherpa-onnx-react-fa-<model>.zip` (each an unzip-and-serve static
site, named after its model). It runs on the `react-fa` branch, like the
Android demo's `apk-fa` workflow.
