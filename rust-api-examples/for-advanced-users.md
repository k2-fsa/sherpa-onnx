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
  `/path/to/sherpa-onnx-v1.13.0-linux-x64-static-lib/lib`

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
| Linux | x86_64 | `sherpa-onnx-v1.13.0-linux-x64-static-lib.tar.bz2` |
| Linux | aarch64 | `sherpa-onnx-v1.13.0-linux-aarch64-static-lib.tar.bz2` |
| macOS | x86_64 | `sherpa-onnx-v1.13.0-osx-x64-static-lib.tar.bz2` |
| macOS | arm64 | `sherpa-onnx-v1.13.0-osx-arm64-static-lib.tar.bz2` |
| Windows | x86_64 | `sherpa-onnx-v1.13.0-win-x64-static-MT-Release-lib.tar.bz2` |

### Shared mode

If you enable the `shared` feature, the build script downloads these shared
archives instead:

| OS | Architecture | Archive example |
|----|--------------|-----------------|
| Linux | x86_64 | `sherpa-onnx-v1.13.0-linux-x64-shared-lib.tar.bz2` |
| Linux | aarch64 | `sherpa-onnx-v1.13.0-linux-aarch64-shared-cpu-lib.tar.bz2` |
| macOS | x86_64 | `sherpa-onnx-v1.13.0-osx-x64-shared-lib.tar.bz2` |
| macOS | arm64 | `sherpa-onnx-v1.13.0-osx-arm64-shared-lib.tar.bz2` |
| Windows | x86_64 | `sherpa-onnx-v1.13.0-win-x64-shared-MT-Release-lib.tar.bz2` |

In practice, use the latest release tag instead of the example version above.

## Configure the `sherpa-onnx` crate in Cargo.toml

### Default configuration

This is enough for most users:

```toml
[dependencies]
sherpa-onnx = "1.13.0"
```

That uses the crate's default feature set.

### Use shared libraries

To use shared libraries instead, disable default features and enable `shared`:

```toml
[dependencies]
sherpa-onnx = { version = "1.13.0", default-features = false, features = ["shared"] }
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

When `SHERPA_ONNX_LIB_DIR` is set, the same behavior applies, but the files come
from your directory instead of an auto-downloaded archive.
