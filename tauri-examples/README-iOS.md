# Using sherpa-onnx with Tauri on iOS

This guide explains how to use the `sherpa-onnx` Rust crate in a Tauri v2 iOS project.

## Overview

On iOS, sherpa-onnx provides pre-built xcframeworks that Xcode links at build
time.  Tauri's `bundle.ios.frameworks` in `tauri.conf.json` tells Xcode where
to find them.

`build.rs` downloads the xcframework(s) automatically during
`cargo tauri ios build` — no manual download is needed.

There are **three** xcframework variants:

| Variant | Archive name | Contents | Extra dependencies |
|---------|-------------|----------|-------------------|
| **static** | `sherpa-onnx-v{ver}-ios-static.xcframework.zip` | Static `.a` (all sherpa-onnx libs merged) | **Requires** `onnxruntime.xcframework` |
| **shared (onnxruntime static)** | `sherpa-onnx-v{ver}-ios-shared-onnxruntime-static.xcframework.zip` | Shared dylib, onnxruntime baked in | None |
| **shared** | `sherpa-onnx-v{ver}-ios-shared.xcframework.zip` | Shared dylib only | Requires separate shared `onnxruntime.xcframework` |

The **shared-onnxruntime-static** variant is the simplest: one xcframework, no
extra dependencies.  The **static** variant produces the smallest binary since
everything is statically linked.

## Quick start (shared)

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

3. Build:

   ```bash
   cargo tauri ios init
   cargo tauri ios build --no-sign
   ```

   `build.rs` downloads `sherpa-onnx-v{ver}-ios-shared-onnxruntime-static.xcframework.zip`
   automatically.

## Quick start (static)

1. Add `sherpa-onnx` to your `src-tauri/Cargo.toml` (default features = static):

   ```toml
   [dependencies]
   sherpa-onnx = "1.13.6"
   ```

2. Tell Tauri to link **both** xcframeworks in `src-tauri/tauri.conf.json`:

   ```json
   {
     "bundle": {
       "iOS": {
         "frameworks": [
           "sherpa-onnx.xcframework",
           "onnxruntime.xcframework"
         ]
       }
     }
   }
   ```

3. Build:

   ```bash
   cargo tauri ios init
   cargo tauri ios build --no-sign
   ```

   `build.rs` downloads both `sherpa-onnx-v{ver}-ios-static.xcframework.zip` and
   `onnxruntime-ios-static-xcframework-{ver}.xcframework.zip` automatically.

## How it works

`build.rs` in `sherpa-onnx-sys` determines the link mode from Cargo features:

1. Downloads the appropriate pre-built xcframework(s) from GitHub Releases into
   the Cargo target cache (`target/.../sherpa-onnx-prebuilt/`).
2. Copies them to the Tauri project root (`src-tauri/`) so that Xcode can find
   them via `bundle.ios.frameworks`.

For static builds, the onnxruntime xcframework is downloaded from
`csukuangfj/onnxruntime-libs` on GitHub.

## Adding to `.gitignore`

The xcframeworks are large binary files.  Add them to `src-tauri/.gitignore`:

```
*.xcframework
```

## CI example (GitHub Actions)

```yaml
- name: Init iOS
  run: cargo tauri ios init

- name: Build iOS
  run: cargo tauri ios build --no-sign
```

No manual xcframework download step is needed — `build.rs` handles it.
