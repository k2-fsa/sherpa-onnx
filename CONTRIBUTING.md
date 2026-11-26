# Contributing to sherpa-onnx

Read this before you write code, and use §13 as the checklist before you push.
Rules are imperative: **MUST** / **NEVER** mean a change will be sent back.

The repository is squash-merged: the PR title becomes the commit subject on
`master`, so title it `Add ...` / `Fix ...` / `Update ...`.

## 1. Adding or changing an option

- **Prefer a per-stream option over a new config field.** A field costs the C
  API, cxx-api, Python, Java, Kotlin, Swift, Rust, Go, C#, Pascal,
  Dart/Flutter, JS/WASM and Node bindings; a stream option costs none.

  ```cpp
  const float length_reward = ss[b]->GetOptionFloat("length_reward", 0);
  const int32_t n = stream->GetOptionInt("max_new_tokens", config.max_new_tokens);
  ```

- **MUST mirror a config field in all of these**, in the same PR:

  | layer | file |
  |---|---|
  | C++ config | `sherpa-onnx/csrc/<model>-model-config.{h,cc}` |
  | C API | `sherpa-onnx/c-api/c-api.h` + `c-api.cc` |
  | C++ wrapper | `sherpa-onnx/c-api/cxx-api.h` + `cxx-api.cc` |
  | Python | `sherpa-onnx/python/csrc/…`, `python/sherpa_onnx/…` |
  | Java/Kotlin/JNI | `java-api/`, `kotlin-api/`, `jni/` |
  | others | Swift, Rust, Go, C#, Pascal, Dart, Node, WASM wrappers |
  | examples + CI | the matching `*-api-examples/` and workflow |

- **C API rules** (ABI):
  - **MUST** append new struct fields at the **end**; never reorder, rename or
    remove existing fields.
  - **NEVER** put `std::string`, `std::vector`, references, templates or
    exceptions in `c-api.h`. C only: `const char *`, plain structs,
    `int32_t`/`int64_t`, pointers.
  - **MUST** document for every new function/field: who owns each pointer,
    whether the function is thread-safe, whether it may block, what `nullptr`
    means.
  - **MUST** add a `HasOption`-style query if users need to know whether an
    option was set.
  - **NEVER** expose an internal C++ struct through the C header.

- **MUST read values from the model** (cache shape, vocab size, layer count)
  instead of adding a flag that can disagree with the graph.

- **MUST document the option** in the header and in `--help`: meaning, units,
  default, valid range, behaviour when out of range.

## 2. Adding a model

New files, not edits to a sibling model:

```
sherpa-onnx/csrc/<model>-model-config.{h,cc}      # Register/Validate/ToString
sherpa-onnx/csrc/<model>-model.{h,cc}             # sessions, Forward*(), caches
sherpa-onnx/csrc/offline-recognizer-<model>-impl.{h,cc}
sherpa-onnx/csrc/<model>-...-test.cc              # no model files needed
```

and edit:

```
sherpa-onnx/csrc/CMakeLists.txt                   # sources + sherpa_onnx_test_srcs
sherpa-onnx/csrc/offline-model-config.{h,cc}      # field, Register, Validate, ToString
sherpa-onnx/csrc/offline-recognizer-impl.cc       # BOTH Create(config) and Create(mgr, config)
sherpa-onnx/c-api/c-api.{h,cc}, cxx-api.{h,cc}
sherpa-onnx/python/csrc/…, python/sherpa_onnx/…   # pybind + from_<model>()
c-api-examples/, cxx-api-examples/, python-api-examples/
```

- **MUST** implement the `Manager *` overload too (`Init()` shared by both
  constructors); Android/HarmonyOS read models from assets.
- **MUST** add the example to `cxx-api-examples/CMakeLists.txt` and a CI step
  that builds and runs it on a downloaded model.
- **MUST** validate at load time: input/output counts, layer count, cache
  shapes, and ONNX `model_type`. Fail with a message that says how to fix it.
- **MUST** include a pure-logic GoogleTest registered in
  `sherpa_onnx_test_srcs`; it must not need model files.

## 3. Reusing existing code

- **MUST** check `sherpa-onnx/csrc/math.h` before writing numeric loops:
  `TopkIndex`, `LogSoftmax`, `Softmax`, `ApplySoftmax`, ...
- **MUST** use Eigen for matrix/energy math; one Eigen expression instead of
  nested loops.
- **MUST** reuse `utils.h`, `text-utils.h` (`Trim`, `SplitString`),
  `file-utils.h` (`ReadFile`, `FileExists`), `onnx-utils.h` (`Clone`,
  `GetInputNames`, `GetOutputNames`), `session.h` (`GetSessionOptions`).
- **MUST** put offline/online shared logic in one helper (`math.h/cc` or
  `utils.h/cc`), not duplicated in two impls.
- **MUST** implement in `.cc`; headers declare, they do not implement
  (templates and tiny inline helpers excepted).
- **MUST** symlink a shared demo file instead of copying it.
- **NEVER** extend an existing model's example for a different model; create a
  new file/folder.

## 4. Memory and ownership

- **NEVER** pass a temporary to `c_str()`:
  `f(std::to_string(x).c_str())` is a use-after-free. Store it in a local.
- **MUST** use RAII for `Ort::Session`, `Ort::Allocator`, file handles,
  buffers. No raw `new`/`delete` or `malloc`/`free`.
- **MUST** free big buffers early (`buf_conv.clear()` or a `{}` scope) to cut
  peak memory.
- **MUST** document borrowed pointers: "not owned, must outlive the stream".
- **MUST** avoid per-frame/per-token allocations in decode loops.
- **MUST** prefer `const &` / `std::move` over copies; avoid `.flatten()` when
  a view works.

## 5. Error handling and invalid input

| situation | behaviour |
|---|---|
| bad config at construction | `SHERPA_ONNX_LOGE(...)` + `SHERPA_ONNX_EXIT(-1)` |
| bad argument at run time | log a warning, return an empty result, keep running |
| optional feature unsupported by the model | log a warning, ignore the option, continue |
| missing file | check at construction, include the path in the message |
| invariant | `SHERPA_ONNX_CHECK` / `SHERPA_ONNX_DCHECK`, not `assert()` |
| exceptions | none; replace `throw`/`Ort::ThrowOnError` with a log + fallback |

- **MUST** `if (n == 0) return {};` and warn when input is too short.
- **MUST** cap unbounded inputs (enrolment audio, hotwords, prompt length) and
  document the cap.
- **NEVER** write `unsigned x; if (x < 0)`; prefer `int32_t`/`int64_t` in
  interfaces.
- **MUST** zero-initialise C structs in C examples (`memset(&s, 0, sizeof s)`).
- **MUST** log values next to names, and quote strings `'%s'` so trailing
  spaces are visible.

## 6. Naming, comments and documentation

- **MUST** use one name per concept (`keyword`/`keywords`, `lm_scale`, file and
  class names consistent).
- **MUST** match the sibling feature's naming and parameter order.
- **MUST** update `README.md` and the CLI `--help` in the same PR when you add
  or change an option.
- **MUST** document: defaults, valid ranges, units, ownership, thread safety,
  whether a call blocks, and model limits (languages, sample rate, window).
- **MUST** keep comments true; a comment describing old behaviour is a bug.
- **MUST** write comments in English and only for non-obvious *why*.
- **MUST** replace the copyright line with your own when copying a template.
- **NEVER** leave commented-out code or an un-issued `TODO`.

## 7. Style and formatting

- **MUST** run `./scripts/check_style_cpplint.sh 1` before pushing. cpplint
  2.0.2 checks **whole files**: if the file you touched already had findings,
  fix them in the same PR.
- **MUST** keep lines ≤ 80 characters, no indentation inside a namespace, no
  blank line after `private:`, no trailing whitespace, sorted includes,
  `int32_t` instead of `int` in interfaces.
- **MUST** run `clang-format` (C/C++), `google-java-format`/IDE formatter
  (Java/Kotlin), a formatter for Python/JS/Dart.
- **MUST** sort CMake source lists alphabetically.
- **NEVER** commit debug prints, `.orig`/`.rej`, editor backups, non-English
  comments or unrelated reformatting.

## 8. Tests, examples and CI

- **MUST** add a test for new logic: `sherpa-onnx/csrc/<name>-test.cc`, in
  `sherpa_onnx_test_srcs`, run with `ctest -R <name> --output-on-failure`.
- **MUST** add a CI step that downloads the model (release `asr-models`
  artifact or HuggingFace) and runs the example.
- **MUST** add an export workflow job for every new conversion script (see
  §10): it exports, verifies, packages and publishes the model.
- **MUST** run it locally before asking for review; a PR that was never
  executed is spotted immediately.
- **MUST** rebase on `master` before review; stale branches fail CI for
  unrelated reasons.
- **MUST** state in the PR description what the numbers are (accuracy, RTF,
  size, CI time) and the commands used.

## 9. Workflows

- **MUST** be `workflow_dispatch`-able.
- **NEVER** gate on the repository owner (`github.repository_owner`) or put it
  in image names; pin the account (`ghcr.io/csukuangfj/...`) so the workflow
  runs in any fork.
- **MUST** pin action versions (`@v4`, or a SHA for third-party actions) and
  Docker images.
- **MUST** use Python `"3.10"` (quoted) in `setup-python` matrices.
- **MUST** mark steps that need secrets (`HF_TOKEN`, release tokens)
  `continue-on-error: true`.
- **MUST** `export` a variable inside a step if that step needs it;
  `"$GITHUB_ENV"` only affects later steps.
- **NEVER** prepend a toolchain `bin/` that ships its own `python3`.
- **MUST** test with the released model artifact, not a personal mirror.
- **MUST** run the workflow in your own fork and link the run in the PR.

## 10. Conversion and verification scripts

- **MUST** ship a `workflow_dispatch` export job with the script: export,
  verify, collect the files and publish. A script nobody can run in CI is
  half a contribution.
- **MUST** keep the verification script small and dependency-light:
  `kaldi-native-fbank` for features, `librosa` for audio, `tokenizers` for
  text, `onnxruntime` + `numpy` for the graph. **NEVER** require
  `transformers` just to run a model.
- **MUST** read audio in one line — `librosa.load(path, sr=16000, mono=True)` —
  instead of hand-written WAV parsing/resampling, and **MUST** truncate to the
  model's window before counting tokens.
- **MUST** turn off randomness (`dither = 0`, fixed seed) so two runs agree.
- **MUST** take dimensions (cache size, token counts) from the exported graph
  or its `metadata.json`, never from a hard-coded constant that can drift.
- **MUST** assert in the script (cosine threshold, argmax equality, expected
  transcript) and exit non-zero on regression.
- **MUST** install every dependency of the script in the workflow's `pip
  install` line (a script that only runs on your machine is not finished).

## 11. Models, export and quantization

- **MUST** add ONNX metadata to every graph: `model_type`, `version`,
  `maintainer`, `url`, `license`, plus dimensions. The loader checks
  `model_type`.
- **MUST** ship **two** complete model directories, fp32 and int8:

  ```
  sherpa-onnx-<model>-<version>/        sherpa-onnx-<model>-<version>-int8/
    *.onnx                                *.int8.onnx
    README.md    # what it is, source, usage   README.md
    LICENSE      # upstream license            LICENSE
    tokenizer/   # vocab/merges/config/...     tokenizer/
    test_wavs/   # en.wav zh.wav fr.wav de.wav ja.wav ko.wav cantonese.wav ...
  ```
  Pack both as `.tar.bz2`, attach to the `asr-models` release, push to
  HuggingFace; the publish steps are `continue-on-error`.
- **MUST** pin the upstream revision and disable dither/randomness in export
  scripts; print torch/transformers/opset versions.
- **MUST** include a `run-onnx.py`/`test.py` that verifies the exported graph
  (cosine + transcript) and fails on regression.
- **MUST** verify quantization (fp32 vs int8 cosine and transcript); keep
  sensitive layers (projector, down-projects, adaLN) in fp32.
- **MUST** document external `.onnx.data` files, or avoid them when the model
  ships to mobile assets.
- **MUST** keep opset compatible with the ONNX Runtime used in CI.

## 12. Build system and dependencies

- **MUST** register new sources in `sherpa-onnx/csrc/CMakeLists.txt` and new
  examples in `cxx-api-examples/CMakeLists.txt`.
- **MUST** add third-party deps through `cmake/<dep>.cmake` (`FetchContent` +
  SHA256), not a `git clone` inside a script.
- **MUST** link extra libraries a provider needs and document the requirement.
- **MUST** run at least one cross build when touching platform code
  (`build-*.sh`: Android, iOS, WASM, aarch64, riscv64, RKNN, ...).

## 13. Before you push

```bash
cmake --build build --target sherpa-onnx-offline <your-target> -j
ctest --test-dir build --output-on-failure -R '<your-test>'
./build/bin/sherpa-onnx-offline --<model>-encoder=... test.wav
./scripts/check_style_cpplint.sh 1
cmake --build build --target _sherpa_onnx -j
python3 python-api-examples/<your-example>.py ...
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/<f>.yaml'))"
bash -n .github/scripts/<script>.sh
```

Then scan the diff:

```bash
git diff master --stat                       # only intended files?
git diff master | grep -nE "TODO|FIXME|std::cout|print\(|debug|/Users/|/home/"
git diff master | grep -nE "your own info|Xiaomi Corporation"   # copyright
```

## 14. PR description

Answer five questions; a PR that does is reviewed quickly:

1. **What** changed (list new files for a new model).
2. **Why** — issue/discussion, model source and license.
3. **Verified how** — commands + numbers table (accuracy, RTF, size, CI time).
4. **Bindings** — which languages were updated, which are follow-ups.
5. **Not covered** — and the plan.

## 15. Review etiquette

- Reply to every comment with the commit that fixes it (`Fixed in a1b2c3d`).
- Push fixes as new commits during review; squash only after the reviewer
  agrees. Do not force-push mid-review.
- If you disagree, give the constraint and an alternative — reasoned push-back
  is accepted.
- "The failure is unrelated" is only acceptable with a link to the run and the
  same failure on `master`.
