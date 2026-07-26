# @nexuscloud/sherpa-onnx

Fork build of [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) with ZipVoice **`espeakVoice`** support — configurable eSpeak language for out-of-lexicon words (e.g. `"si"` for Sinhala).

Built from [ishankrishan/sherpa-onnx](https://github.com/ishankrishan/sherpa-onnx) via GitHub Actions (`.github/workflows/build.yaml`).

## Contents

| Path | Purpose |
|------|---------|
| `sherpa-onnx-wasm-nodejs.js` + `.wasm` | Electron / Node desktop |
| `sherpa-onnx-*.js` | JS bindings (`espeakVoice` in ZipVoice config) |
| `android/jniLibs/{abi}/*.so` | Capacitor Android native engine |
| `android/kotlin-api/Tts.kt` | Kotlin OfflineTts API |
| `scripts/sync-android.mjs` | Sync Android bits into your Capacitor app |

## Install

```bash
npm install @nexuscloud/sherpa-onnx
```

## Sync Android (Capacitor)

From your app root:

```bash
node node_modules/@nexuscloud/sherpa-onnx/scripts/sync-android.mjs --force
```

## ZipVoice espeakVoice (desktop)

```javascript
const tts = createOfflineTts({
  model: {
    zipvoice: {
      // ...
      espeakVoice: 'si',  // Sinhala OOV phonemization
    },
  },
});
```

## Local validation (before publish)

```bash
# Keep CommonJS TTS binding in sync with browser source
python3 wasm/tts/sync-tts-node.py

# Fail on syntax errors (catches stray braces after hand-edits)
node --check wasm/tts/sherpa-onnx-tts.js
node --check wasm/tts/sherpa-onnx-tts.node.js
node --check scripts/nodejs/index.js
```

CI (`build.yaml`) runs the same checks before WASM/Android builds and again
on the assembled npm package so a bad JS file cannot be published.

## License

Apache-2.0 (same as upstream sherpa-onnx)
