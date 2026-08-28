# Using sherpa-onnx with Tauri on iOS

This guide explains how to use the `sherpa-onnx` Rust crate in a Tauri v2 iOS project.

## Overview

On iOS, sherpa-onnx uses **shared linking only**. The pre-built xcframework
contains a shared dylib (`SherpaOnnxC.framework/SherpaOnnxC`) with onnxruntime
statically linked in.

## Prerequisites

- macOS with Xcode installed
- [Tauri CLI](https://v2.tauri.app/start/prerequisites/): `cargo install tauri-cli`

## Quick start

1. Add `sherpa-onnx` to your `src-tauri/Cargo.toml`:

   ```toml
   [dependencies]
   sherpa-onnx = { version = "1.13.6", default-features = false, features = ["shared"] }
   ```

2. Tell Tauri to link the xcframework in `src-tauri/tauri.conf.json`:

   ```json
   {
     "bundle": {
       "iOS": {
         "frameworks": ["sherpa-onnx.xcframework"]
       }
     }
   }
   ```

3. Run the setup script to download the xcframework (first build only):

   ```bash
   src-tauri/setup-ios.sh
   ```

4. Build:

   ```bash
   cargo tauri ios init
   cargo tauri ios build --target aarch64 --no-sign          # device
   cargo tauri ios build --target aarch64-sim --no-sign       # simulator
   ```

   After the first build, `build.rs` caches the xcframework and subsequent
   builds are automatic.

## How it works

### Why is `setup-ios.sh` needed?

Xcode checks for xcframework existence **before** running any build phases.
`build.rs` downloads the xcframework during the build (inside the "Build Rust
Code" script phase), which is too late — Xcode has already failed.

`setup-ios.sh` bridges this gap by downloading the xcframework before the
first build. It is idempotent — if the xcframework already exists, it exits
immediately.

### What `build.rs` does

`build.rs` in `sherpa-onnx-sys` handles subsequent builds:

1. Downloads `sherpa-onnx-v{ver}-ios-shared-onnxruntime-static.xcframework.zip`
   from GitHub Releases into the Cargo target cache.
2. Copies the xcframework to the Tauri project root (`src-tauri/`) so that
   Xcode can find it via `bundle.ios.frameworks`.

### Adding to `.gitignore`

The xcframework is a large binary. Add it to `src-tauri/.gitignore`:

```
*.xcframework
```

## CI example (GitHub Actions)

```yaml
- name: Download pre-built iOS xcframework
  shell: bash
  run: |
    cd tauri-examples/hello_world
    src-tauri/setup-ios.sh

- name: Init iOS
  run: |
    cd tauri-examples/hello_world
    cargo tauri ios init

- name: Build iOS
  run: |
    cd tauri-examples/hello_world
    cargo tauri ios build --target aarch64 --no-sign
```
