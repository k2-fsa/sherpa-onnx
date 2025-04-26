# Sherpa-ONNX WASM Combined Module

This directory contains the WebAssembly (WASM) combined module for Sherpa-ONNX, which includes support for:
- Automatic Speech Recognition (ASR)
- Voice Activity Detection (VAD)
- Text-to-Speech (TTS)
- Keyword Spotting (KWS)
- Speaker Diarization
- Speech Enhancement

## File Structure

When built, the following files are generated:
- `sherpa-onnx-wasm-combined.js` - The main JavaScript glue code
- `sherpa-onnx-wasm-combined.wasm` - The WebAssembly binary
- `sherpa-onnx-wasm-combined.data` - The preloaded assets (models)
- JS library files:
  - `sherpa-onnx-combined-core.js` - Core functionality
  - `sherpa-onnx-combined-asr.js` - ASR functionality
  - `sherpa-onnx-combined-vad.js` - VAD functionality
  - `sherpa-onnx-combined-tts.js` - TTS functionality
  - `sherpa-onnx-combined-kws.js` - Keyword Spotting functionality
  - `sherpa-onnx-combined-speaker.js` - Speaker Diarization functionality
  - `sherpa-onnx-combined-enhancement.js` - Speech Enhancement functionality
  - `sherpa-onnx-combined.js` - Combined functionality wrapper

## Building

To build the WASM module:

```bash
cd /path/to/sherpa-onnx
./build-wasm-combined.sh
```

This script will:
1. Create a `build-wasm-combined` directory
2. Configure CMake with the necessary options
3. Build the WASM module
4. Install the files to `bin/wasm/combined`
5. Copy the files to the original repo at `wasm/combined`

## Important Notes

1. **Large Asset Bundle**: The `.data` file can be very large (300MB+) as it contains all preloaded models. For production, consider using dynamic loading of models instead.

2. **File Locations**: All files must be in the same directory for the WASM module to work correctly. The `.data` file MUST be in the same directory as the `.js` and `.wasm` files.

3. **Local Testing**: To test locally, run a web server from the `wasm/combined` directory:

```bash
cd /path/to/sherpa-onnx/wasm/combined
python -m http.server 8000
```

Then open `http://localhost:8000` in your browser.

## License

Same as Sherpa-ONNX. 