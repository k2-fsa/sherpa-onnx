# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

sherpa-onnx: offline/streaming ASR, TTS, VAD, speaker diarization/ID, KWS, punctuation, speech enhancement, etc., running on ONNX Runtime (plus RKNN/QNN/Axera/Ascend/etc. backends). The core is C++; every other language binding wraps it.

## Build

CMake project; the root `CMakeLists.txt` options are `SHERPA_ONNX_*` (e.g. `SHERPA_ONNX_ENABLE_TESTS`, `SHERPA_ONNX_ENABLE_PYTHON`, `SHERPA_ONNX_ENABLE_GPU`, `SHERPA_ONNX_ENABLE_TTS`, `SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION`, `BUILD_SHARED_LIBS`). `build/` is the local build dir (tests and Python are OFF in it by default).

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DSHERPA_ONNX_ENABLE_TESTS=ON ..
make -j$(nproc)
```

- Cross-compile / platform builds are the top-level `build-*.sh` scripts (android, ios, wasm, aarch64, riscv64, rknn, ...).
- Python wheel: `python setup.py install` (or `pip install .`); `SHERPA_ONNX_CMAKE_ARGS` passes extra cmake flags.
- Tests are GoogleTest, built only with `SHERPA_ONNX_ENABLE_TESTS=ON`. Each `*-test.cc` becomes its own executable named after the file, registered with ctest. Add new test files to `sherpa_onnx_test_srcs` in `sherpa-onnx/csrc/CMakeLists.txt`.
  ```bash
  cd build && ctest -R circular-buffer-test --output-on-failure   # single test
  ./bin/circular-buffer-test --gtest_filter='Suite.Case'            # single case (check build/bin or build/sherpa-onnx/csrc)
  ```
- Python tests: `sherpa-onnx/python/tests/` (pytest style; some need downloaded models).
- Lint: cpplint 2.0.2 via `./scripts/check_style_cpplint.sh` (no arg = last commit, `1` = uncommitted changes, `2` = whole repo). Config in `CPPLINT.cfg`; CI is `.github/workflows/style_check.yaml`. Headers use `SHERPA_ONNX_CSRC_<NAME>_H_` guards and a `// sherpa-onnx/csrc/<file>` + copyright header comment.

## Architecture

- `sherpa-onnx/csrc/` — all core C++ (~600 files), plus the CLI executables (`sherpa-onnx.cc`, `sherpa-onnx-offline.cc`, `*-microphone*.cc`, websocket servers, ...). `csrc/README.md` lists the binaries.
- Language bindings are thin layers over core, and a new config field/feature usually has to be mirrored in each: `sherpa-onnx/c-api` (C API + `cxx-api`), `sherpa-onnx/python/csrc` (pybind11) and `python/sherpa_onnx`, `sherpa-onnx/jni` + `java-api` + `kotlin-api`, `rust`, `pascal-api`, and the wrappers in `swift-api-examples`, `dotnet-examples`, `go-api-examples`, `nodejs-addon-examples`, `flutter`, `dart-api-examples`, `harmony-os`, `wasm`. Each language has a `*-examples` dir that CI tests.
- **Model-dispatch pattern**: public facades (`OnlineRecognizer`, `OfflineRecognizer`, `OfflineTts`, ...) hold a `*Impl` created by `*Impl::Create(config)` in `*-impl.cc`, which inspects the config / model metadata to pick a concrete implementation (`offline-recognizer-<model>-impl.h`, `online-recognizer-{transducer,ctc,paraformer}-impl.h`). Each model family has its own `*-model.{h,cc}`, `*-model-config.{h,cc}` and `*-model-meta-data.h`. Adding a model = new model + config + impl + registration in the dispatcher + `sources` in `csrc/CMakeLists.txt` + `Config` plumbing in every binding.
- Configs are structs with `Register(ParseOptions*)`, `Validate()`, `ToString()`; CLI flags come from `Register`, and bindings expose the same fields.
- VAD: `VadModel` (`silero-vad-model`, `ten-vad-model`) behind `VoiceActivityDetector`, which segments speech using a `CircularBuffer`.
- `cmake/` holds per-dependency fetch scripts (onnxruntime, kaldi-native-fbank, etc.); dependencies are downloaded at configure time.
- `scripts/` contains per-model export/conversion scripts (not part of the build); `.github/workflows/` has the large CI matrix (per-platform builds, per-language tests, model export).
