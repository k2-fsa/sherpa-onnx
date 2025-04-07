# Sherpa-ONNX Combined WebAssembly Module

This directory contains a combined WebAssembly module for the Sherpa-ONNX project, which integrates multiple features:

- Automatic Speech Recognition (ASR)
- Voice Activity Detection (VAD)
- Text-to-Speech Synthesis (TTS)
- Speech Enhancement
- Speaker Diarization
- Keyword Spotting

## How to Use

### Loading the Module

You can use the combined module in two ways:

#### Option 1: Load Individual Modules (Recommended)

This approach loads only the components you need:

```html
<!-- First load the WASM module -->
<script src="sherpa-onnx-wasm-combined.js"></script>

<!-- Load the core module which is required by all other modules -->
<script src="sherpa-onnx-core.js"></script>

<!-- Then load only the modules you need -->
<script src="sherpa-onnx-vad.js"></script>
<!-- Add other modules as needed -->

<script>
  // This callback is called when the WASM module is loaded
  window.onModuleReady = function() {
    // Your initialization code here
    console.log("Module ready!");
  };
</script>
```

#### Option 2: Load All Modules via the Combined Loader

This approach loads all available modules:

```html
<!-- First load the WASM module -->
<script src="sherpa-onnx-wasm-combined.js"></script>

<!-- Then load the combined module that will load all other modules -->
<script src="sherpa-onnx-combined.js"></script>

<script>
  // This callback is called when all modules are loaded
  window.onSherpaOnnxReady = function() {
    // Your initialization code here
    console.log("All modules loaded!");
  };
</script>
```

### Module Structure

The codebase has been organized into modular files:

- `sherpa-onnx-core.js`: Core functionality, utilities, and file system operations
- `sherpa-onnx-vad.js`: Voice Activity Detection functionality
- `sherpa-onnx-combined.js`: Loader that loads all individual modules

Additional modules will be added in the future:
- `sherpa-onnx-asr.js`: Automatic Speech Recognition functionality
- `sherpa-onnx-tts.js`: Text-to-Speech functionality
- And more...

## Demo Application

The included `index.html` demonstrates how to use the combined module. It shows:

1. How to load models from URLs
2. How to initialize each component (ASR, VAD, TTS)
3. How to stream audio from the microphone
4. How to get results from each component

## Building the Module

The WebAssembly module can be built using the provided build script:

```bash
cd /path/to/sherpa-onnx
./build-wasm-combined.sh
```

The built files will be located in `bin/wasm/combined/` and are also copied to `wasm/combined/`.

## Setting Up Models

Before using the demo, you need to set up model files:

```bash
cd /path/to/sherpa-onnx/wasm/combined
./setup-assets.sh
```

This script will download necessary model files to the `assets/` directory.

## Troubleshooting

- **Module load errors**: Ensure the WASM module is loaded before any other scripts
- **Model load errors**: Check the browser console for specific error messages
- **Audio capture issues**: Make sure your browser has permission to access the microphone
- **Performance issues**: Try reducing buffer sizes or using smaller models 