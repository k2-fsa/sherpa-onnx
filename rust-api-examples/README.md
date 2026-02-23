# Introduction

This folder uses Rust API maintained by us.

## Setup library path

### Method 1 (Build from source)

```bash
export SHERPA_ONNX_LIB_DIR=/Users/fangjun/open-source/sherpa-onnx/build/install/lib
export RUSTFLAGS="-C link-arg=-Wl,-rpath,$SHERPA_ONNX_LIB_DIR"
```

### Method 2 (Download pre-built libs)

```bash
# You can choose any directory you like
cd $HOME/Downloads

# We use version v1.12.25 below as an example.
# Please always use the latest version from
# https://github.com/k2-fsa/sherpa-onnx/releases

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.12.25/sherpa-onnx-v1.12.25-osx-universal2-shared.tar.bz2
tar xvf sherpa-onnx-v1.12.25-osx-universal2-shared.tar.bz2
rm sherpa-onnx-v1.12.25-osx-universal2-shared.tar.bz2

export SHERPA_ONNX_LIB_DIR=$HOME/Downloads/sherpa-onnx-v1.12.25-osx-universal2-shared/lib
export RUSTFLAGS="-C link-arg=-Wl,-rpath,$SHERPA_ONNX_LIB_DIR"
```

## Run it

```bash
cargo run --example version
```

For macOS, you can run
```
otool -l target/debug/examples/version | grep -A2 LC_RPATH
```
to check the RPATH.

# Alternative rust bindings for sherpa-onnx

Please see also to https://github.com/thewh1teagle/sherpa-rs
