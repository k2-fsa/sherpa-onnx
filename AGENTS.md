# AGENTS.md

Guidance for AI agents (Claude Code, Codex, Cursor, ...) working in this
repository. The first part is identical to `CLAUDE.md`; the last section is a
short rulebook distilled from the review history.

> Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a PR: the file
> checklist for a new model, the API/binding surface, error-handling and
> memory rules, style, CI/workflow conventions, and the pre-push check list.

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

## Working as an agent in this repo

`CONTRIBUTING.md` is the rulebook; read it before coding. The rules that matter
most, in order:

1. **Prefer a per-stream option** (`SetOption`/`GetOptionInt`) over a new config
   field — a field costs the C API, cxx-api, Python, Java, Kotlin, Swift, Rust,
   Go, C#, Pascal, Dart, WASM and Node bindings.
2. **C API**: append fields at the end, no `std::string` in the header, document
   ownership and thread safety, add a `HasOption` query, never expose internal
   structs.
3. **Reuse** `math.h` (`TopkIndex`, `LogSoftmax`, Eigen), `utils.h`,
   `text-utils.h`, `file-utils.h`, `onnx-utils.h`; implement in `.cc`, keep
   headers for declarations; new models get new files; add the `Manager *`
   overload.
4. **Memory**: no temporaries in `c_str()`, RAII for ORT/handles, free big
   buffers early, no per-frame allocation, document borrowed pointers.
5. **Errors**: `SHERPA_ONNX_EXIT(-1)` only at construction; at run time log
   with `SHERPA_ONNX_LOGE` and return an empty result; warn and ignore
   unsupported options; `SHERPA_ONNX_CHECK` instead of `assert`; no exceptions.
6. **Bounds**: `if (n == 0) return {};`, warn on short input, cap unbounded
   inputs, never check `unsigned < 0`.
7. **Naming/docs**: one name per concept, document every option (default,
   range, units, ownership, thread safety), update `README.md` and `--help`,
   English comments, replace the copyright line.
8. **Style**: `./scripts/check_style_cpplint.sh 1` (whole files, 80 columns,
   sorted includes), `clang-format`, no commented-out code, no debug output.
9. **Tests/CI**: a GoogleTest that needs no model files, a CI step that runs
   the example on a downloaded model, an export `workflow_dispatch` job for
   every conversion script, run everything locally, rebase on `master`.
10. **Workflows**: `workflow_dispatch`-able, no `repository_owner` gating, pin
    action versions, Python `"3.10"` quoted, `continue-on-error` for steps that
    need secrets, run it in your own fork.
11. **PR**: one topic, only intended files, no drive-by reformatting, remove
    temp/debug files, describe the problem + commands + measured numbers.
12. **Models/scripts**: ONNX metadata, dimensions read from the graph, two
    complete model directories (fp32 + int8: README, LICENSE, tokenizer, test
    wavs) packed as `.tar.bz2`, verified quantization, pinned upstream
    revisions; verification scripts use kaldi-native-fbank/librosa/tokenizers
    and assert on cosine/transcript.

Do not leave local paths, machine names, credentials or fork-specific URLs in
the code or in the workflows.
