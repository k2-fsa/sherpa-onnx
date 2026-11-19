# Forced-alignment Web app (WASM)

Two pre-built web demos are available, one app per model: a **web demo**
built from [`wasm/forced-alignment`](../../wasm/forced-alignment) and a
full-featured **web app** built from
[`react-examples/forced-alignment`](../../react-examples/forced-alignment).

## Web demo (wasm/forced-alignment)

The files `sherpa-onnx-v<version>-wasm-fa-<model>.zip` of the current release, e.g.
[v2.0.2](https://github.com/k2-fsa/sherpa-onnx/releases/tag/v2.0.2), contain
the plain-web demo: unzip a file and open `index.html` with any static HTTP
server. It covers the basics — pick an audio/video file or record from the
microphone, enter the transcript, align, play words and export the result.

## Web app (react-examples/forced-alignment)

The files `sherpa-onnx-react-fa-<model>.zip` of the
[react-fa](https://github.com/k2-fsa/sherpa-onnx/releases/tag/react-fa)
release contain the full-featured web app described below. The remainder of
this page refers to it.

## What the app does

A pure-browser forced-alignment tool (Chinese UI): pick an audio/video file or
record from the microphone, enter the transcript, and get the start/end time
of every word. Click a row to play just that word (sample-accurate), export a
Praat **TextGrid** or txt files, and share the result with other apps
(WeChat, ...). Everything runs locally in the browser — no backend, no
uploads, offline-capable.

## Try it in your browser

The apps are hosted on GitHub Pages — click a link to open the demo of the
model you want (the first visit downloads the model, 3–37 MB, with a
progress bar):

| Model | Language | Link |
|-------|----------|------|
| `zh-aishell3-pinyin` | Chinese | [Open the demo](https://k2-fsa.github.io/sherpa/onnx/forced-alignment/demo/zh-aishell3-pinyin/) |
| `zh-biaobei-pinyin` | Chinese | [Open the demo](https://k2-fsa.github.io/sherpa/onnx/forced-alignment/demo/zh-biaobei-pinyin/) |
| `zh-large-pinyin` | Chinese | [Open the demo](https://k2-fsa.github.io/sherpa/onnx/forced-alignment/demo/zh-large-pinyin/) |
| `english-mfa-v2.2.1` | English | [Open the demo](https://k2-fsa.github.io/sherpa/onnx/forced-alignment/demo/english-mfa-v2.2.1/) |
| `english-mfa-v3.0.0` | English | [Open the demo](https://k2-fsa.github.io/sherpa/onnx/forced-alignment/demo/english-mfa-v3.0.0/) |
| `english-mfa-v3.1.0` | English | [Open the demo](https://k2-fsa.github.io/sherpa/onnx/forced-alignment/demo/english-mfa-v3.1.0/) |

The entry points in full:

- `zh-aishell3-pinyin` (Chinese):
  https://k2-fsa.github.io/sherpa/onnx/forced-alignment/demo/zh-aishell3-pinyin/
- `zh-biaobei-pinyin` (Chinese):
  https://k2-fsa.github.io/sherpa/onnx/forced-alignment/demo/zh-biaobei-pinyin/
- `zh-large-pinyin` (Chinese):
  https://k2-fsa.github.io/sherpa/onnx/forced-alignment/demo/zh-large-pinyin/
- `english-mfa-v2.2.1` (English):
  https://k2-fsa.github.io/sherpa/onnx/forced-alignment/demo/english-mfa-v2.2.1/
- `english-mfa-v3.0.0` (English):
  https://k2-fsa.github.io/sherpa/onnx/forced-alignment/demo/english-mfa-v3.0.0/
- `english-mfa-v3.1.0` (English):
  https://k2-fsa.github.io/sherpa/onnx/forced-alignment/demo/english-mfa-v3.1.0/

## Which file do I need?

One app per model. Download the zip of the model you want, unzip it, and
serve the directory with any static HTTP server (see
[the README](../../react-examples/forced-alignment/README.md) for hosting
notes).

| File | Language | Model | Notes |
|------|----------|-------|-------|
| `sherpa-onnx-react-fa-zh-aishell3-pinyin.zip` | Chinese | [zh-aishell3-pinyin](https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/zh-aishell3-pinyin.tar.bz2) | pinyin lexicon, trained on AISHELL-3 |
| `sherpa-onnx-react-fa-zh-biaobei-pinyin.zip` | Chinese | [zh-biaobei-pinyin](https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/zh-biaobei-pinyin.tar.bz2) | pinyin lexicon, trained on the biaobei reading corpus |
| `sherpa-onnx-react-fa-zh-large-pinyin.zip` | Chinese | [zh-large-pinyin](https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/zh-large-pinyin.tar.bz2) | pinyin lexicon, trained on AISHELL-3 + biaobei + more |
| `sherpa-onnx-react-fa-english-mfa-v2.2.1.zip` | English | [english-mfa-v2.2.1](https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/english-mfa-v2.2.1.tar.bz2) | MFA `english_mfa` acoustic model v2.2.1, 100 phones, ~95k-word dictionary |
| `sherpa-onnx-react-fa-english-mfa-v3.0.0.zip` | English | [english-mfa-v3.0.0](https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/english-mfa-v3.0.0.tar.bz2) | MFA `english_mfa` acoustic model v3.0.0, 100 phones, ~95k-word dictionary |
| `sherpa-onnx-react-fa-english-mfa-v3.1.0.zip` | English | [english-mfa-v3.1.0](https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/english-mfa-v3.1.0.tar.bz2) | MFA `english_mfa` acoustic model v3.1.0, 100 phones, 42k-word core dictionary |

The Chinese models are interchangeable; `zh-large` is the most robust,
`zh-aishell3`/`zh-biaobei` match their training domain. For English, prefer
`en-v3.1.0` unless you need the larger dictionary of v2.2.1/v3.0.0 (see
[README.md](README.md)).

The source code of the app is in
[`react-examples/forced-alignment`](../../react-examples/forced-alignment)
(the wasm module is in [`wasm/forced-alignment`](../../wasm/forced-alignment)).

## How to download and run it locally

1. Download the zip of your model from the table above, e.g.
   `sherpa-onnx-react-fa-english-mfa-v3.1.0.zip` from the
   [react-fa](https://github.com/k2-fsa/sherpa-onnx/releases/tag/react-fa)
   release.
2. Unzip it: `unzip sherpa-onnx-react-fa-<model>.zip` — this creates a
   `sherpa-onnx-react-fa-<model>/` folder with the whole app (html, css, js
   and the wasm package with the model inside).
3. Serve the folder with any static HTTP server:

   ```bash
   cd sherpa-onnx-react-fa-<model>
   python3 -m http.server 8000
   ```

4. Open a browser and visit <http://localhost:8000>. The model loads with a
   progress bar on first visit (the model is inside the wasm data file; the
   first load downloads it, later loads come from the browser cache).

Any other static HTTP server works as well (nginx, `npx serve`, ...); the
only requirement is that `.wasm` files are served as `application/wasm`.

## How to build it yourself

See
[react-examples/forced-alignment/README.md](../../react-examples/forced-alignment/README.md):
build the wasm module with `./build-wasm-simd-forced-alignment.sh` (put the
model package into `wasm/forced-alignment/assets/` first), then
`./build-static.sh <model_index>`. The CI does exactly this via
[`.github/workflows/react-fa.yaml`](../../.github/workflows/react-fa.yaml).
