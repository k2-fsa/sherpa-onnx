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

### Example 1: Show sherpa-onnx version

```bash
cargo run --example version
```

For macOS, you can run
```
otool -l target/debug/examples/version | grep -A2 LC_RPATH
```
to check the RPATH.

### Example 2: ASR with streaming zipformer (with a file)

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2

tar xvf sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2
rm sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2

cargo run --example streaming_zipformer -- \
    --wav sherpa-onnx-streaming-zipformer-en-2023-06-21/test_wavs/1.wav \
    --encoder sherpa-onnx-streaming-zipformer-en-2023-06-21/encoder-epoch-99-avg-1.int8.onnx \
    --decoder sherpa-onnx-streaming-zipformer-en-2023-06-21/decoder-epoch-99-avg-1.onnx \
    --joiner sherpa-onnx-streaming-zipformer-en-2023-06-21/joiner-epoch-99-avg-1.int8.onnx \
    --tokens sherpa-onnx-streaming-zipformer-en-2023-06-21/tokens.txt \
    --provider cpu \
    --debug
```

### Example 3: ASR with streaming zipformer (with a microphone, real-time ASR)

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
rm sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2

cargo run --example streaming_zipformer_microphone --features mic -- \
    --encoder sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.int8.onnx \
    --decoder sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx \
    --joiner sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.int8.onnx \
    --tokens sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt \
    --provider cpu \
    --debug
```

### Example 4: ASR with non-streaming SenseVoice

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2

tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2
rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2
ls -lh sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17

cargo run --example sense_voice -- \
    --wav ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/test_wavs/en.wav \
    --model ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/model.int8.onnx \
    --tokens ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/tokens.txt
```

# Alternative rust bindings for sherpa-onnx

Please see also https://github.com/thewh1teagle/sherpa-rs
