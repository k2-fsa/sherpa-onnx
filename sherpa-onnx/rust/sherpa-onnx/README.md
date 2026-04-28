# sherpa-onnx Rust crate

Safe Rust bindings for the public `sherpa-onnx` inference APIs.

## Setup

For most users, this is enough:

```toml
[dependencies]
sherpa-onnx = "1.13.0"
```

The default Rust configuration uses **static** linking.

If `SHERPA_ONNX_LIB_DIR` is not set, the build script automatically downloads a
matching prebuilt native `-lib` archive from
[GitHub releases](https://github.com/k2-fsa/sherpa-onnx/releases) and uses it
for the build.
The downloader also honors standard proxy environment variables such as
`HTTPS_PROXY`, `HTTP_PROXY`, and `NO_PROXY` (including lowercase variants).

## Use shared libraries

To use **shared** libraries instead of the default static mode:

```toml
[dependencies]
sherpa-onnx = { version = "1.13.0", default-features = false, features = ["shared"] }
```

When shared libraries are used:

- Linux and macOS binaries are built with runtime search paths for both the
  original library directory and the executable directory
- the build script also copies the required shared runtime libraries next to the
  generated Cargo binaries/examples
- Windows copies the required DLLs next to the generated binaries/examples

That means shared mode should work for normal users without manually setting
`LD_LIBRARY_PATH` or `DYLD_LIBRARY_PATH`.

## Use your own libraries

If you already have native sherpa-onnx libraries, set:

```bash
export SHERPA_ONNX_LIB_DIR=/path/to/sherpa-onnx/lib
```

That override works for both static and shared builds. If it is set, the build
script uses your directory instead of auto-downloading another archive.

## More examples

See:

- [Rust examples README](https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/README.md)
- [Rust examples advanced guide](https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/for-advanced-users.md)
