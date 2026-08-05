# Testing Flutter Web Support for sherpa-onnx

## Prerequisites

- Emscripten SDK installed (version 4.0.23 recommended)
- Flutter SDK installed (>= 3.24.0)
- Chrome browser

## Step 1: Build the WASM module for web

```bash
cd /path/to/sherpa-onnx

# Build the WASM module targeting browser (not Node.js)
./build-wasm-simd-web.sh
```

This produces:
- `build-wasm-simd-web/install/bin/wasm/web/sherpa-onnx-wasm-web.js`
- `build-wasm-simd-web/install/bin/wasm/web/sherpa-onnx-wasm-web.wasm`

## Step 2: Copy WASM assets to the Flutter web plugin

```bash
./build-flutter-web-wasm.sh
```

Or manually:

```bash
cp build-wasm-simd-web/install/bin/wasm/web/sherpa-onnx-wasm-web.js \
   flutter/sherpa_onnx_web/assets/

cp build-wasm-simd-web/install/bin/wasm/web/sherpa-onnx-wasm-web.wasm \
   flutter/sherpa_onnx_web/assets/
```

## Step 3: Run the hello_world example in Chrome

```bash
cd flutter-examples/hello_world
flutter pub get
flutter run -d chrome
```

The app should display:
- sherpa-onnx version
- Git SHA1
- Git date
- onnxruntime version

## Step 4: Build for deployment (optional)

```bash
cd flutter-examples/hello_world
flutter build web
```

Output is in `build/web/`. Serve with any HTTP server:

```bash
cd build/web
python3 -m http.server 8080
```

Open http://localhost:8080 in Chrome.

## Architecture

```
sherpa-onnx C API
    ↓ (Emscripten compiles to WASM)
sherpa-onnx-wasm-web.js + .wasm
    ↓ (loaded by sherpa_onnx_web plugin)
wasm_ffi DynamicLibrary
    ↓ (conditional import: ffi_proxy.dart)
Dart bindings (sherpa_onnx_bindings.dart)
    ↓
Flutter app (hello_world)
```

## Key files

| File | Purpose |
|------|---------|
| `wasm/wasm-common.cmake` | Shared exported functions (104) and flags |
| `wasm/web/CMakeLists.txt` | Browser-specific WASM build config |
| `build-wasm-simd-web.sh` | Build WASM for browser |
| `build-flutter-web-wasm.sh` | Build + copy to Flutter plugin |
| `flutter/sherpa_onnx_web/` | Flutter web plugin (loads WASM via wasm_ffi) |
| `flutter/sherpa_onnx/lib/src/ffi_proxy.dart` | Conditional: dart:ffi on native, wasm_ffi on web |
| `flutter/sherpa_onnx/lib/src/init_native.dart` | Native library loading (dart:io) |
| `flutter/sherpa_onnx/lib/src/init_web.dart` | Web library stub (set by sherpa_onnx_web) |

## How it works

1. `build-wasm-simd-web.sh` compiles the sherpa-onnx C API to WebAssembly using Emscripten. Unlike the nodejs build, it does NOT use `-sNODERAWFS=1` (Node-specific filesystem) so it works in browsers.

2. `flutter/sherpa_onnx_web` is a Flutter web plugin that:
   - Calls `Memory.init()` from wasm_ffi to set up WASM memory
   - Uses `inject_js` to load the Emscripten JS glue code
   - Uses `rootBundle.load()` to fetch the .wasm binary
   - Calls `EmscriptenModule.compile()` to instantiate the WASM module
   - Registers the resulting `DynamicLibrary` with `init_web.dart`

3. `flutter/sherpa_onnx/lib/src/ffi_proxy.dart` uses a conditional export:
   - On native: exports `dart:ffi`
   - On web: exports `package:wasm_ffi/wasm_ffi.dart`

4. All 20+ Dart source files in `lib/src/` import `ffi_proxy.dart` instead of `dart:ffi` directly, so the same bindings work on both native and web.

5. `init_native.dart` uses `dart:io` (Platform detection) to load the native library. `init_web.dart` expects the WASM module to be pre-loaded by `SherpaOnnxWeb.loadWasm()`.

## Troubleshooting

### "WASM module not loaded" error

Make sure `SherpaOnnxWeb.loadWasm()` is called before `initBindingsAsync()`.
See `flutter-examples/hello_world/lib/main.dart` for the correct order.

### Chrome console shows "SharedArrayBuffer is not defined"

The WASM build does not use pthreads, so SharedArrayBuffer is not required.
If you see this error, it means you're using the old pthread-enabled build.
Rebuild with `./build-wasm-simd-web.sh`.

### WASM file not found (404)

Make sure the assets are copied to `flutter/sherpa_onnx_web/assets/`:

```bash
ls flutter/sherpa_onnx_web/assets/sherpa-onnx-wasm-web.wasm
```

### Build fails with "SHERPA_ONNX_ENABLE_WASM_WEB" error

Make sure you're running `./build-wasm-simd-web.sh` (not the nodejs script).
The web build uses `-DSHERPA_ONNX_ENABLE_WASM_WEB=ON`.
