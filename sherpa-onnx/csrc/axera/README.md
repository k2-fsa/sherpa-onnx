# ZipVoice TTS on AX650

Build with `SHERPA_ONNX_ENABLE_AXERA=ON` and select `--provider=axera`.
ZipVoice uses the existing text frontend, reference-audio processing, C/C++ API,
and generation callbacks. Its encoder, four decoder partitions, and Vocos run
through AX_ENGINE. Duration expansion, Euler updates, and inverse STFT run on
the CPU. No Python inference process is required.

This backend supports the AX650 four-part exports from
[AXERA-TECH/ZipVoice.AXERA](https://huggingface.co/AXERA-TECH/ZipVoice.AXERA).
It validates tensor names, counts, dtypes, shapes, and sizes before use. Other
hardware exports or different static shapes are not supported by this backend.

## Build in an ARM64 Linux container

On an Apple Silicon Mac, use a Linux ARM64 container with the same OS/compiler
as the board. The validation build uses Ubuntu 22.04, GCC 11.4, and CMake 3.22.
Copy the board's SDK headers and libraries into `axera-sdk/include` and
`axera-sdk/lib` before building. Replace `BOARD` with your SSH alias:

```bash
mkdir -p axera-sdk build-axera
ssh BOARD 'tar -C /soc -czf - include lib' | tar -C axera-sdk -xzf -
docker run --rm --platform linux/arm64 \
  -v "$PWD:/src:ro" \
  -v "$PWD/axera-sdk:/opt/axera:ro" \
  -v "$PWD/build-axera:/build" \
  ubuntu:22.04 bash -ceu '
    apt-get update
    apt-get install -y g++ cmake ninja-build curl git python3 pkg-config
    export SHERPA_ONNX_AXERA_LIB_DIR=/opt/axera/lib
    cmake -S /src -B /build -G Ninja \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O3 -DNDEBUG" \
      -DCMAKE_C_FLAGS_RELWITHDEBINFO="-O3 -DNDEBUG" \
      -DCMAKE_CXX_FLAGS=-I/opt/axera/include \
      -DBUILD_SHARED_LIBS=ON \
      -DSHERPA_ONNX_ENABLE_AXERA=ON \
      -DSHERPA_ONNX_ENABLE_TTS=ON \
      -DSHERPA_ONNX_ENABLE_TESTS=ON \
      -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
      -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
      -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
      -DSHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION=OFF
    cmake --build /build --target sherpa-onnx-offline-tts sherpa-onnx-c-api \
      offline-tts-zipvoice-length-test ax-engine-guard-test -j8
    cp -a /build/_deps/onnxruntime-src/lib/libonnxruntime* /build/lib/
  '
```

This uses optimized code without the project's Release-only LTO pass. Deploy
`build-axera/bin/` and `build-axera/lib/` to the board and run the tests there;
AX_ENGINE requires the board's NPU driver. Keep the copied SDK outside Git.

## Optional native build

Install a C++17 compiler, CMake, and the board's AXERA SDK/runtime. With the SDK
headers in `/soc/include` and libraries in `/soc/lib`:

```bash
export SHERPA_ONNX_AXERA_LIB_DIR=/soc/lib
cmake -S . -B build-axera \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS=-I/soc/include \
  -DBUILD_SHARED_LIBS=ON \
  -DSHERPA_ONNX_ENABLE_AXERA=ON \
  -DSHERPA_ONNX_ENABLE_TTS=ON \
  -DSHERPA_ONNX_ENABLE_TESTS=ON \
  -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
  -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF
cmake --build build-axera --target \
  sherpa-onnx-offline-tts offline-tts-zipvoice-length-test \
  ax-engine-guard-test -j4
ctest --test-dir build-axera -R offline-tts-zipvoice-length-test \
  --output-on-failure
export LD_LIBRARY_PATH="$PWD/build-axera/lib:/soc/lib:${LD_LIBRARY_PATH:-}"
```

The AX650N validation device uses Ubuntu 22.04 ARM64 and SDK
`V3.6.2_20250603154858`. AXERA remains disabled by default. CPU-only builds do
not require the AXERA SDK.

## Download models and frontend resources

Install the Hugging Face CLI (`hf`) on a machine with network access, then:

```bash
hf download AXERA-TECH/ZipVoice.AXERA \
  --revision aa48d14426f528a63ccb0720edb79651597e5316 \
  --include 'models/zipvoice_distill_ax650/*' \
  --include 'models/zipvoice_ax650/*' \
  --include 'models/vocoder/vocos_full.axmodel' \
  --include 'resources/zipvoice_hf/zipvoice/tokens.txt' \
  --include 'assets/moss_prompts/zh_1_4p5s.wav' \
  --include 'assets/moss_prompts/en_4_4p5s.wav' \
  --include 'assets/paragraphs/*' \
  --local-dir zipvoice-axera

curl -L --fail \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2 \
  -o frontend.tar.bz2
tar xf frontend.tar.bz2
```

Only `lexicon.txt` and `espeak-ng-data/` from the second archive are needed by
the AXERA backend. The archive's ONNX encoder/decoder are used only for CPU
inference. The AXERA and sherpa-onnx packages have identical `tokens.txt` files
at the tested revision. Keep model weights and generated audio outside Git.

## Generate a Chinese sentence

```bash
frontend=./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia
models=./zipvoice-axera
./build-axera/bin/sherpa-onnx-offline-tts \
  --provider=axera \
  --num-threads=1 \
  --zipvoice-encoder="$models/models/zipvoice_distill_ax650/encoder.axmodel" \
  --zipvoice-decoder="$models/models/zipvoice_distill_ax650/decoder4_split_manifest.json" \
  --zipvoice-vocoder="$models/models/vocoder/vocos_full.axmodel" \
  --zipvoice-tokens="$models/resources/zipvoice_hf/zipvoice/tokens.txt" \
  --zipvoice-lexicon="$frontend/lexicon.txt" \
  --zipvoice-data-dir="$frontend/espeak-ng-data" \
  --zipvoice-guidance-scale=3.0 \
  --num-steps=4 \
  --reference-audio="$models/assets/moss_prompts/zh_1_4p5s.wav" \
  --reference-text='不管怎么样我和汤姆还是要感谢贝尔卡金的援手' \
  --output-filename=zipvoice-axera-zh.wav \
  '今天午后天气很好，我打开窗户，听见远处有人聊天，水杯也轻轻晃了一下。'
```

`zipvoice.decoder` refers to the JSON manifest for `provider=axera`; partition
filenames are resolved relative to that manifest. For `provider=cpu`, it keeps
its existing meaning as an ONNX decoder filename. The public C API structures
and language-binding configuration fields are unchanged.

For the standard model, select `models/zipvoice_ax650/encoder.axmodel` and the
manifest in that directory, use `--num-steps=10`, and set
`--zipvoice-guidance-scale=1.0`. Sampling parameters are explicit; the backend
does not automatically override them with `runtime_config.json`.

For English, use `en_4_4p5s.wav`, reference text
`This is almost twice the current industry production level per train.`, and
an English target sentence. The existing espeak-ng frontend handles English;
the AXERA demo's simplified character tokenizer and Python helper are not used.
For a paragraph, pass `"$(cat "$models/assets/paragraphs/zh_ginkgo.txt")"` as the
text argument. Generated chunks are delivered through the existing callback.

## Capacity and runtime behavior

* Tokens are int32 IDs in the published vocabulary range 0–359, concatenated
  as reference + target + padding, at most 384.
* Decoder tensors are `[1,1024,100]`; padding masks are uint8 `[1,1024]`.
  Distill uses manifest version 1. Standard uses version 2 and forwards the
  batch-2 hidden state, uint8 `padding_mask2`, and scalar `cfg_scale` between
  partitions for classifier-free guidance.
* Vocos accepts `[1,100,620]` mel and returns real/imaginary spectra. Only valid
  frames are passed to inverse STFT, producing 24 kHz mono audio.
* The reference and generated mel together must fit 1024 frames; generated mel
  must fit 620 frames. The frontend splits text using token counts, actual
  reference-frame count, and requested speed. It never silently clips duration.
* A 4.5-second reference takes about 422 frames, leaving at most 602 generated
  frames (about 6.41 seconds per chunk). Longer references reduce this budget.
  Empty, silent, oversized, or otherwise unusable references fail generation.
* Models and physical IO buffers stay resident for the TTS object's lifetime.
  Generation on an object is serialized. Output caches are invalidated before
  CPU access. AX_SYS/AX_ENGINE initialization is shared across threads and model
  objects and is released after the final guard is destroyed.
* Cancellation keeps the existing callback behavior between generated chunks;
  an individual AX_ENGINE_RunSync call cannot be interrupted by that callback.

The standalone vendor demo's synthesis RTF excludes model loading and frontend
work. Measure the sherpa CLI's elapsed time separately; do not compare it with
that narrower metric as if both represented complete request latency.

An optional hardware regression checks that a context survives destruction of
another thread's final runtime guard:

```bash
export SHERPA_ONNX_AXERA_TEST_ENCODER="$PWD/zipvoice-axera/models/zipvoice_distill_ax650/encoder.axmodel"
ctest --test-dir build-axera -R ax-engine-guard-test --output-on-failure
```

The test skips when that environment variable is unset.

## Validation

The container-built binaries were tested on AX650N with the public reference
WAVs and paragraph texts from the pinned model repository. Distill used four
steps and guidance 3.0; standard used ten steps and guidance 1.0. All six
generation cases produced nonempty 24 kHz mono WAVs.

| Distill case | Audio duration (s) | CLI generation time (s) | RTF |
| --- | ---: | ---: | ---: |
| Chinese sentence above | 5.833 | 1.683 | 0.289 |
| Chinese paragraph `zh_ginkgo.txt` | 38.432 | 8.375 | 0.218 |
| English sentence | 6.211 | 2.150 | 0.346 |
| English paragraph `en_scavenger.txt` | 65.363 | 16.744 | 0.256 |

The English sentence was `This morning, a small train left the station,
carrying sleepy passengers toward a bright coastal town.` Standard was smoke
tested with `你好，欢迎使用语音合成。` and `Welcome to speech synthesis.`;
standard long paragraphs were not tested in this integration run. CLI
generation time excludes model construction; complete distill CLI process
times were 3.236, 9.387, 3.180, and 17.742 seconds respectively. These are
individual smoke measurements, not a controlled benchmark or speech-quality
evaluation.

Silent and oversized references were rejected. A C API smoke check verified
finite float samples, cancellation after the first callback, and reuse of the
same TTS object after cancellation and invalid input. An ONNX CPU-provider
smoke generation passed, as did compilation of the affected dispatch/frontend
translation units with AXERA disabled and SDK includes removed.

On the board, run the regression binaries directly after deployment:

```bash
export LD_LIBRARY_PATH="$PWD/build-axera/lib:/soc/lib:${LD_LIBRARY_PATH:-}"
./build-axera/bin/offline-tts-zipvoice-length-test
SHERPA_ONNX_AXERA_TEST_ENCODER="$PWD/zipvoice-axera/models/zipvoice_distill_ax650/encoder.axmodel" \
  ./build-axera/bin/ax-engine-guard-test
```

Written by LittleMouse.
