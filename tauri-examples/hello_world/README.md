# Tauri hello world

This is a minimal Tauri app that displays sherpa-onnx version information:

- sherpa-onnx version
- Git SHA1
- Git date
- Onnxruntime version

It supports **Linux, macOS, Windows, iOS, and Android**.

## Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) (stable)
- [Node.js](https://nodejs.org/) (for the Tauri CLI)
- Platform-specific Tauri prerequisites: see [Tauri prerequisites guide](https://v2.tauri.app/start/prerequisites/)

## Build from source (using local sherpa-onnx)

```bash
# From the repository root
cd tauri-examples/hello_world

# Use local sherpa-onnx crate (optional, for development)
mkdir -p src-tauri/.cargo
cat > src-tauri/.cargo/config.toml <<'EOF'
[patch.crates-io]
sherpa-onnx = { path = "../../../sherpa-onnx/rust/sherpa-onnx" }
sherpa-onnx-sys = { path = "../../../sherpa-onnx/rust/sherpa-onnx-sys" }
EOF

npm install
npm run build
```

The built app will be in `src-tauri/target/release/bundle/`.

## Run in development mode

```bash
cd tauri-examples/hello_world
npm install
npm run dev
```

## Running a downloaded app on macOS

If you download the `.zip` from GitHub Releases and see:

> "hello_world" is damaged and can't be opened. You should move it to the Trash.

This is macOS Gatekeeper blocking an unsigned app. Fix it with:

```bash
xattr -cr hello_world.app
```

Then double-click `hello_world.app` to run it.

## Build for specific platforms

### Desktop (Linux, macOS, Windows)

```bash
npm run build
```

### iOS

```bash
# Requires Xcode and an iOS device/simulator
npm run tauri ios init
npm run tauri ios build
```

### Android

```bash
# Requires Android Studio and SDK
npm run tauri android init
npm run tauri android build
```
