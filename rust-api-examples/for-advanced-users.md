# Rust examples: advanced users

This note is for users who want to control how the `sherpa-onnx` Rust crate
finds and links its native libraries.

Most users do **not** need anything here. The default behavior is:

1. build normally
2. let the build script download the matching native libraries automatically
3. run the examples or your own Rust program

## Use your own sherpa-onnx libraries

If you already have sherpa-onnx libraries on disk, set
`SHERPA_ONNX_LIB_DIR` to the `lib` directory before building:

```bash
export SHERPA_ONNX_LIB_DIR=/path/to/sherpa-onnx/lib
```

Examples:

- source build output: `/path/to/sherpa-onnx/build/install/lib`
- manually extracted release archive:
  `/path/to/sherpa-onnx-v1.13.7-linux-x64-static-lib/lib`

If `SHERPA_ONNX_LIB_DIR` is set, the build script uses that directory and does
not auto-download another archive.

## Automatic download behavior

If `SHERPA_ONNX_LIB_DIR` is not set, `sherpa-onnx-sys/build.rs` downloads a
matching prebuilt `-lib` archive from GitHub releases and uses its `lib`
directory automatically.

The build script currently selects archives like this:

### Default mode

Default mode uses the default crate feature set, which means **static** linking.
Most users just get this behavior automatically.

| OS | Architecture | Archive example |
|----|--------------|-----------------|
| Linux | x86_64 | [`sherpa-onnx-v1.13.7-linux-x64-static-lib.tar.bz2`](https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.7/sherpa-onnx-v1.13.7-linux-x64-static-lib.tar.bz2) |
| Linux | aarch64 | [`sherpa-onnx-v1.13.7-linux-aarch64-static-lib.tar.bz2`](https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.7/sherpa-onnx-v1.13.7-linux-aarch64-static-lib.tar.bz2) |
| macOS | x86_64 | [`sherpa-onnx-v1.13.7-osx-x64-static-lib.tar.bz2`](https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.7/sherpa-onnx-v1.13.7-osx-x64-static-lib.tar.bz2) |
| macOS | arm64 | [`sherpa-onnx-v1.13.7-osx-arm64-static-lib.tar.bz2`](https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.7/sherpa-onnx-v1.13.7-osx-arm64-static-lib.tar.bz2) |
| Windows | x86_64 | [`sherpa-onnx-v1.13.7-win-x64-static-MT-Release-lib.tar.bz2`](https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.7/sherpa-onnx-v1.13.7-win-x64-static-MT-Release-lib.tar.bz2) |
| Windows | arm64 | [`sherpa-onnx-v1.13.7-win-arm64-static-MT-Release-lib.tar.bz2`](https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.7/sherpa-onnx-v1.13.7-win-arm64-static-MT-Release-lib.tar.bz2) |

### Shared mode

If you enable the `shared` feature, the build script downloads these shared
archives instead:

| OS | Architecture | Archive example |
|----|--------------|-----------------|
| Linux | x86_64 | [`sherpa-onnx-v1.13.7-linux-x64-shared-lib.tar.bz2`](https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.7/sherpa-onnx-v1.13.7-linux-x64-shared-lib.tar.bz2) |
| Linux | aarch64 | [`sherpa-onnx-v1.13.7-linux-aarch64-shared-cpu-lib.tar.bz2`](https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.7/sherpa-onnx-v1.13.7-linux-aarch64-shared-cpu-lib.tar.bz2) |
| macOS | x86_64 | [`sherpa-onnx-v1.13.7-osx-x64-shared-lib.tar.bz2`](https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.7/sherpa-onnx-v1.13.7-osx-x64-shared-lib.tar.bz2) |
| macOS | arm64 | [`sherpa-onnx-v1.13.7-osx-arm64-shared-lib.tar.bz2`](https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.7/sherpa-onnx-v1.13.7-osx-arm64-shared-lib.tar.bz2) |
| Windows | x86_64 | [`sherpa-onnx-v1.13.7-win-x64-shared-MT-Release-lib.tar.bz2`](https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.7/sherpa-onnx-v1.13.7-win-x64-shared-MT-Release-lib.tar.bz2) |
| Windows | arm64 | [`sherpa-onnx-v1.13.7-win-arm64-shared-MT-Release-lib.tar.bz2`](https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.7/sherpa-onnx-v1.13.7-win-arm64-shared-MT-Release-lib.tar.bz2) |
| iOS | arm64 | [`sherpa-onnx-v1.13.7-ios-shared-onnxruntime-static.xcframework.zip`](https://github.com/k2-fsa/sherpa-onnx/releases/download/xcframework/sherpa-onnx-v1.13.7-ios-shared-onnxruntime-static.xcframework.zip) |
| Android | all ABIs | [`sherpa-onnx-v1.13.7-android.tar.bz2`](https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.7/sherpa-onnx-v1.13.7-android.tar.bz2) |

In practice, use the latest release tag instead of the example version above.

## Configure the `sherpa-onnx` crate in Cargo.toml

### Default configuration

This is enough for most users:

```toml
[dependencies]
sherpa-onnx = "1.13.7"
```

That uses the crate's default feature set.

### Use shared libraries

To use shared libraries instead, disable default features and enable `shared`:

```toml
[dependencies]
sherpa-onnx = { version = "1.13.7", default-features = false, features = ["shared"] }
```

From the command line, the equivalent example command is:

```bash
cargo run --no-default-features --features shared --example version
```

### Enable microphone examples

In `rust-api-examples`, microphone support is controlled by the `mic` feature:

```bash
cargo run --features mic --example streaming_zipformer_microphone -- --help
```

If you want both microphone support and shared libraries:

```bash
cargo run --no-default-features --features "shared,mic" \
  --example streaming_zipformer_microphone -- --help
```

## Notes about runtime behavior

When shared libraries are used:

- Linux and macOS: the build script adds both absolute and relative runtime
  rpath entries automatically
- Linux and macOS: the build script also copies the required shared runtime
  libraries next to Cargo-generated binaries and examples
- Windows: the build script copies the required DLLs next to the generated
  binaries automatically
- iOS: the build script copies the required shared runtime libraries next to
  Cargo-generated binaries (same behavior as macOS)

When `SHERPA_ONNX_LIB_DIR` is set, the same behavior applies, but the files come
from your directory instead of an auto-downloaded archive.

## Cross-compiling for iOS

iOS supports real devices (`aarch64-apple-ios`) and simulators
(`aarch64-apple-ios-sim`, `x86_64-apple-ios`). iOS only supports **shared**
linking (the build script auto-selects shared mode for iOS, like Android).

### Prerequisites

- macOS with Xcode installed
- Rust iOS target: `rustup target add aarch64-apple-ios`
- Tauri CLI: `cargo install tauri-cli`

### Build with Tauri

The easiest way to build an iOS app with sherpa-onnx is through
[Tauri v2](https://v2.tauri.app/). See the
[hello_world example](../tauri-examples/hello_world/) for a working setup.

```bash
cd tauri-examples/hello_world
npm install
cargo tauri ios init
cargo tauri ios build
```

The build script automatically downloads the iOS xcframework archive and creates
symlinks so the linker finds the libraries under the expected names.

### Build for iOS

```bash
cargo build --target aarch64-apple-ios --release
```

This downloads the
`sherpa-onnx-v{VERSION}-ios-shared-onnxruntime-static.xcframework.zip` archive
from the
[xcframework release tag](https://github.com/k2-fsa/sherpa-onnx/releases/tag/xcframework).
The shared dylib has onnxruntime statically linked in, so no separate onnxruntime
framework is needed.

### Use your own iOS xcframework

If you already have an iOS xcframework, set `SHERPA_ONNX_LIB_DIR` to a directory
containing `libsherpa-onnx-c-api.dylib`:

```bash
export SHERPA_ONNX_LIB_DIR=/path/to/ios-dylib-dir
cargo build --target aarch64-apple-ios --release
```

Alternatively, set `SHERPA_ONNX_ARCHIVE_DIR` to a directory containing the
`.xcframework.zip` archive, and the build script will use it instead of
downloading:

```bash
export SHERPA_ONNX_ARCHIVE_DIR=/path/to/archives
# The build script looks for sherpa-onnx-v{VERSION}-ios-shared-onnxruntime-static.xcframework.zip
# in this directory.
```

### iOS archive structure

The iOS xcframework archive is published on the
[xcframework release tag](https://github.com/k2-fsa/sherpa-onnx/releases/tag/xcframework)
(not the versioned release tag):

`sherpa-onnx-v{VERSION}-ios-shared-onnxruntime-static.xcframework.zip`

After extraction, the xcframework contains:

```
SherpaOnnxC.xcframework/
  ios-arm64/
    SherpaOnnxC.framework/
      SherpaOnnxC          # shared dylib with onnxruntime statically linked
      Headers/
      Modules/
      Info.plist
  ios-arm64_x86_64-simulator/
    SherpaOnnxC.framework/
      ...
```

The build script finds `ios-arm64/SherpaOnnxC.framework/SherpaOnnxC` and
creates a symlink `libsherpa-onnx-c-api.dylib` in a `lib/` directory so the
Rust linker finds the library under the expected name.

## Cross-compiling for Android

Android support uses **shared** libraries only (static linking is not
supported). The build script downloads a single archive containing all ABIs.

### Supported ABIs

| Android ABI | Rust target | Architecture |
|-------------|-------------|--------------|
| arm64-v8a | `aarch64-linux-android` | 64-bit ARM |
| armeabi-v7a | `armv7-linux-androideabi` | 32-bit ARM |
| x86 | `i686-linux-android` | 32-bit x86 |
| x86_64 | `x86_64-linux-android` | 64-bit x86 |

### Prerequisites

- Android NDK (r27 or later recommended)
- Rust Android targets: `rustup target add aarch64-linux-android armv7-linux-androideabi`
- Set `ANDROID_NDK_HOME` or `ANDROID_NDK_ROOT` to your NDK installation
- Tauri CLI: `cargo install tauri-cli`

### Build with Tauri

The easiest way to build an Android app with sherpa-onnx is through
[Tauri v2](https://v2.tauri.app/). See the
[hello_world example](../tauri-examples/hello_world/) for a working setup.

```bash
cd tauri-examples/hello_world
npm install
cargo tauri android init
cargo tauri android build
```

The build script automatically downloads the Android archive and places the
shared libraries in the correct `jniLibs/{abi}/` directory.

### Build a shared library for Android

```bash
# Example: arm64
cargo build --target aarch64-linux-android --release --no-default-features --features shared

# Example: armv7
cargo build --target armv7-linux-androideabi --release --no-default-features --features shared
```

This downloads `sherpa-onnx-v{VERSION}-android.tar.bz2` from
[GitHub releases](https://github.com/k2-fsa/sherpa-onnx/releases) and extracts
the shared libraries for the target ABI.

### Android archive structure

The Android archive is a single `.tar.bz2` file containing all ABIs:

```
sherpa-onnx-v{VERSION}-android/
  jniLibs/
    arm64-v8a/
      libsherpa-onnx-c-api.so
      libonnxruntime.so
    armeabi-v7a/
      libsherpa-onnx-c-api.so
      libonnxruntime.so
    x86/
      libsherpa-onnx-c-api.so
      libonnxruntime.so
    x86_64/
      libsherpa-onnx-c-api.so
      libonnxruntime.so
```

The build script selects the correct ABI directory based on the Rust target
triple (e.g., `aarch64-linux-android` maps to `arm64-v8a`).

### Use your own Android libraries

If you already have Android shared libraries, set `SHERPA_ONNX_LIB_DIR` to a
directory containing the `.so` files for your target ABI:

```bash
export SHERPA_ONNX_LIB_DIR=/path/to/jniLibs/arm64-v8a
cargo build --target aarch64-linux-android --release --no-default-features --features shared
```

Alternatively, set `SHERPA_ONNX_ARCHIVE_DIR` to a directory containing the
`sherpa-onnx-v{VERSION}-android.tar.bz2` archive:

```bash
export SHERPA_ONNX_ARCHIVE_DIR=/path/to/archives
cargo build --target aarch64-linux-android --release --no-default-features --features shared
```

### Notes

- Android only supports the `shared` feature. Do not use the default `static`
  feature for Android targets.
- The archive includes `libonnxruntime.so` alongside `libsherpa-onnx-c-api.so`.
  Both are required at runtime.
- The build script sets `rpath` to `$ORIGIN` so the shared libraries are found
  relative to the binary at runtime.

## Building on Linux

Linux supports both **static** and **shared** linking on x86_64 and aarch64.

### Supported architectures

| Architecture | Rust target | Archive (static) | Archive (shared) |
|--------------|-------------|------------------|------------------|
| x86_64 | `x86_64-unknown-linux-gnu` | [`sherpa-onnx-v1.13.7-linux-x64-static-lib.tar.bz2`](https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.7/sherpa-onnx-v1.13.7-linux-x64-static-lib.tar.bz2) | [`sherpa-onnx-v1.13.7-linux-x64-shared-lib.tar.bz2`](https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.7/sherpa-onnx-v1.13.7-linux-x64-shared-lib.tar.bz2) |
| aarch64 | `aarch64-unknown-linux-gnu` | [`sherpa-onnx-v1.13.7-linux-aarch64-static-lib.tar.bz2`](https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.7/sherpa-onnx-v1.13.7-linux-aarch64-static-lib.tar.bz2) | [`sherpa-onnx-v1.13.7-linux-aarch64-shared-cpu-lib.tar.bz2`](https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.7/sherpa-onnx-v1.13.7-linux-aarch64-shared-cpu-lib.tar.bz2) |

### Prerequisites

- GCC or Clang
- For aarch64 cross-compilation: `rustup target add aarch64-unknown-linux-gnu`
  and an aarch64 cross-compiler (e.g., `gcc-aarch64-linux-gnu`)

### Build with Tauri

```bash
cd tauri-examples/hello_world
npm install
npm run build
```

### Build with default (static) linking

```bash
cargo build --release
```

This downloads the matching static archive and produces a self-contained binary.
The static build links against `libstdc++`, `libm`, `libpthread`, and `libdl`.

### Build with shared linking

```bash
cargo build --release --no-default-features --features shared
```

This downloads the matching shared archive. The build script:

- sets `rpath` to `$ORIGIN` so `libsherpa-onnx-c-api.so` and
  `libonnxruntime.so` are found next to the binary at runtime
- copies the `.so` files next to the generated binaries and examples

### Cross-compile for aarch64

```bash
# Static
cargo build --target aarch64-unknown-linux-gnu --release

# Shared
cargo build --target aarch64-unknown-linux-gnu --release \
  --no-default-features --features shared
```

### Build from source

If you prefer building sherpa-onnx from source instead of downloading prebuilt
archives:

```bash
mkdir build && cd build
cmake \
  -DBUILD_SHARED_LIBS=ON \
  -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
  -DBUILD_ESPEAK_NG_EXE=OFF \
  -DSHERPA_ONNX_ENABLE_BINARY=OFF \
  -DCMAKE_INSTALL_PREFIX=./install \
  ..
cmake --build . --target install --config Release

export SHERPA_ONNX_LIB_DIR=$PWD/install/lib
cargo build --release --no-default-features --features shared
```

Use `-DBUILD_SHARED_LIBS=OFF` for static linking instead.

### Use your own libraries

```bash
export SHERPA_ONNX_LIB_DIR=/path/to/lib
cargo build --release
```

Or point to a directory containing the release archive:

```bash
export SHERPA_ONNX_ARCHIVE_DIR=/path/to/archives
cargo build --release
```

### Notes

- The shared archives include both `libsherpa-onnx-c-api.so` and
  `libonnxruntime.so`. Both are required at runtime.
- The aarch64 shared archive has a `-cpu` suffix in its name
  (`linux-aarch64-shared-cpu-lib`) to distinguish it from GPU variants.
- For static builds, the binary is self-contained with no runtime `.so`
  dependencies (aside from system libraries).

## Building on Windows

Windows supports both **static** and **shared** linking on x86_64 and arm64.
All Windows builds use the MSVC toolchain with static CRT (`/MT`).

### Supported architectures

| Architecture | Rust target | Archive (static) | Archive (shared) |
|--------------|-------------|------------------|------------------|
| x86_64 | `x86_64-pc-windows-msvc` | [`sherpa-onnx-v1.13.7-win-x64-static-MT-Release-lib.tar.bz2`](https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.7/sherpa-onnx-v1.13.7-win-x64-static-MT-Release-lib.tar.bz2) | [`sherpa-onnx-v1.13.7-win-x64-shared-MT-Release-lib.tar.bz2`](https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.7/sherpa-onnx-v1.13.7-win-x64-shared-MT-Release-lib.tar.bz2) |
| arm64 | `aarch64-pc-windows-msvc` | [`sherpa-onnx-v1.13.7-win-arm64-static-MT-Release-lib.tar.bz2`](https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.7/sherpa-onnx-v1.13.7-win-arm64-static-MT-Release-lib.tar.bz2) | [`sherpa-onnx-v1.13.7-win-arm64-shared-MT-Release-lib.tar.bz2`](https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.7/sherpa-onnx-v1.13.7-win-arm64-shared-MT-Release-lib.tar.bz2) |

### Prerequisites

- Visual Studio 2022 (or later) with the "Desktop development with C++" workload
- Rust MSVC toolchain: `rustup target add x86_64-pc-windows-msvc`
- For arm64: `rustup target add aarch64-pc-windows-msvc`

### Important: select the correct MSVC linker

Cargo on Windows may find Git's `link.exe` before MSVC's. You must explicitly
point Cargo at the MSVC linker by setting the appropriate environment variable:

For x86_64:

```powershell
$env:CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER = "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC\<version>\bin\HostX64\x64\link.exe"
```

For arm64:

```powershell
$env:CARGO_TARGET_AARCH64_PC_WINDOWS_MSVC_LINKER = "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC\<version>\bin\HostX64\ARM64\link.exe"
```

Alternatively, use `ilammy/msvc-dev-cmd@v1` in CI and set the linker from
`$env:VCToolsInstallDir`:

```powershell
$linker = Join-Path $env:VCToolsInstallDir 'bin\HostX64\x64\link.exe'
$env:CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER = $linker
```

### Build with Tauri

```powershell
cd tauri-examples\hello_world
npm install
npm run build
```

### Build with default (static) linking

```powershell
cargo build --release
```

This downloads the matching static archive and produces a self-contained binary.
The static CRT (`/MT`) is used automatically.

### Build with shared linking

```powershell
cargo build --release --no-default-features --features shared
```

This downloads the matching shared archive. The build script copies the required
DLLs next to the generated binaries automatically:

- `sherpa-onnx-c-api.dll`
- `onnxruntime.dll`
- `onnxruntime_providers_shared.dll`

All three DLLs are required at runtime.

### Cross-compile for arm64 from x64

On an x64 Windows host, you can cross-compile for arm64:

```powershell
rustup target add aarch64-pc-windows-msvc

# Set the cross-compilation linker
$env:CARGO_TARGET_AARCH64_PC_WINDOWS_MSVC_LINKER = "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC\<version>\bin\HostX64\ARM64\link.exe"

# Static
cargo build --target aarch64-pc-windows-msvc --release

# Shared
cargo build --target aarch64-pc-windows-msvc --release --no-default-features --features shared
```

### Build from source

```powershell
mkdir build
cd build
cmake -A x64 -DBUILD_SHARED_LIBS=ON -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF -DCMAKE_INSTALL_PREFIX=./install ..
cmake --build . --config Release
cmake --build . --config Release --target install

$env:SHERPA_ONNX_LIB_DIR = "$PWD\install\lib"
cd ..
cargo build --release --no-default-features --features shared
```

Use `-DBUILD_SHARED_LIBS=OFF` for static linking instead. For arm64, replace
`-A x64` with `-A ARM64` and run from a Developer Command Prompt with the
arm64 toolchain.

### Use your own libraries

```powershell
$env:SHERPA_ONNX_LIB_DIR = "C:\path\to\lib"
cargo build --release
```

Or point to a directory containing the release archive:

```powershell
$env:SHERPA_ONNX_ARCHIVE_DIR = "C:\path\to\archives"
cargo build --release
```

### Notes

- All Windows archives use static CRT (`/MT`). Dynamic CRT (`/MD`) variants
  are available on the release page but are not used by the Rust build script.
- The shared archives include `sherpa-onnx-c-api.dll`, `onnxruntime.dll`, and
  `onnxruntime_providers_shared.dll`. All three are required at runtime.
- For static builds, the binary is self-contained with no runtime DLL
  dependencies (aside from system DLLs).
- If you see `link.exe` errors, make sure the MSVC linker is used instead of
  Git's `link.exe`. See the "select the correct MSVC linker" section above.

## Building on macOS

macOS supports both **static** and **shared** linking on arm64 (Apple Silicon)
and x86_64 (Intel).

### Supported architectures

| Architecture | Rust target | Archive (static) | Archive (shared) |
|--------------|-------------|------------------|------------------|
| arm64 | `aarch64-apple-darwin` | [`sherpa-onnx-v1.13.7-osx-arm64-static-lib.tar.bz2`](https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.7/sherpa-onnx-v1.13.7-osx-arm64-static-lib.tar.bz2) | [`sherpa-onnx-v1.13.7-osx-arm64-shared-lib.tar.bz2`](https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.7/sherpa-onnx-v1.13.7-osx-arm64-shared-lib.tar.bz2) |
| x86_64 | `x86_64-apple-darwin` | [`sherpa-onnx-v1.13.7-osx-x64-static-lib.tar.bz2`](https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.7/sherpa-onnx-v1.13.7-osx-x64-static-lib.tar.bz2) | [`sherpa-onnx-v1.13.7-osx-x64-shared-lib.tar.bz2`](https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.7/sherpa-onnx-v1.13.7-osx-x64-shared-lib.tar.bz2) |

### Prerequisites

- Xcode Command Line Tools: `xcode-select --install`
- For x86_64 cross-compilation on arm64: `rustup target add x86_64-apple-darwin`

### Build with Tauri

```bash
cd tauri-examples/hello_world
npm install
npm run build
```

### Build with default (static) linking

```bash
cargo build --release
```

This downloads the matching static archive and produces a self-contained binary.
The static build links against `libc++` and the `Foundation` framework.

### Build with shared linking

```bash
cargo build --release --no-default-features --features shared
```

This downloads the matching shared archive. The build script:

- sets `rpath` to `@loader_path` so `libsherpa-onnx-c-api.dylib` and
  `libonnxruntime.dylib` are found next to the binary at runtime
- copies the `.dylib` files next to the generated binaries and examples

### Cross-compile for x86_64 from arm64

On an Apple Silicon Mac, you can cross-compile for Intel:

```bash
rustup target add x86_64-apple-darwin

# Static
cargo build --target x86_64-apple-darwin --release

# Shared
cargo build --target x86_64-apple-darwin --release \
  --no-default-features --features shared
```

### Build from source

```bash
mkdir build && cd build
cmake \
  -DCMAKE_OSX_ARCHITECTURES="arm64" \
  -DBUILD_SHARED_LIBS=ON \
  -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
  -DBUILD_ESPEAK_NG_EXE=OFF \
  -DSHERPA_ONNX_ENABLE_BINARY=OFF \
  -DCMAKE_INSTALL_PREFIX=./install \
  ..
cmake --build . --target install --config Release

export SHERPA_ONNX_LIB_DIR=$PWD/install/lib
cargo build --release --no-default-features --features shared
```

Use `-DBUILD_SHARED_LIBS=OFF` for static linking instead. For a universal
(arm64 + x86_64) build, use `-DCMAKE_OSX_ARCHITECTURES="arm64;x86_64"`.

### Use your own libraries

```bash
export SHERPA_ONNX_LIB_DIR=/path/to/lib
cargo build --release
```

Or point to a directory containing the release archive:

```bash
export SHERPA_ONNX_ARCHIVE_DIR=/path/to/archives
cargo build --release
```

### Notes

- The shared archives include both `libsherpa-onnx-c-api.dylib` and
  `libonnxruntime.dylib`. Both are required at runtime.
- For static builds, the binary is self-contained with no runtime `.dylib`
  dependencies (aside from system frameworks).
- The `@loader_path` rpath ensures shared libraries are found relative to the
  binary, so no `DYLD_LIBRARY_PATH` setup is needed in most cases.
