# Introduction

This folder contains React (TypeScript + Vite) web app examples for
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx). Each app runs fully in
the browser via WebAssembly — no backend, no uploads.

You can try forced alignment online in your browser, no installation needed:
<https://k2-fsa.github.io/sherpa/onnx/forced-alignment/web.html#try-it-in-your-browser>

| Folder | Pre-built apps | Description |
|--------|----------------|-------------|
| [forced-alignment](./forced-alignment) | [react-fa](https://github.com/k2-fsa/sherpa-onnx/releases/tag/react-fa) | Forced alignment: given an audio/video file (or a microphone recording) and its transcript, it computes the start/end time of every word. See [../scripts/mfa/web.md](../scripts/mfa/web.md). |
