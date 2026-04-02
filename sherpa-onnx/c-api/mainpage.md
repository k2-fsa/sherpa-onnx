# sherpa-onnx public API documentation

This documentation covers the public native APIs shipped in:

- `c-api.h` — the C API
- `cxx-api.h` — the C++ wrapper built on top of the C API

These headers expose the main sherpa-onnx inference features for native
applications and for language bindings that need a stable ABI.

## What is documented here

The generated docs include the public APIs for:

- streaming ASR
- non-streaming ASR
- keyword spotting
- voice activity detection
- offline text-to-speech
- spoken language identification
- speaker embedding extraction and speaker management
- audio tagging
- offline and online punctuation
- linear resampling
- offline speaker diarization
- offline and online speech enhancement

The C API also includes HarmonyOS-specific constructor variants where
applicable.

## Which header should I use?

Use `c-api.h` if you are:

- writing C code
- building FFI bindings for other languages
- integrating through a plain C ABI

Use `cxx-api.h` if you are:

- writing C++ code directly
- preferring RAII wrappers over manual destroy/free calls
- preferring `std::string`, `std::vector`, and move-only wrapper classes

## Common ownership rules

For the C API:

- objects created by `SherpaOnnxCreate*()` are usually destroyed with a
  matching `SherpaOnnxDestroy*()`
- result snapshots, returned strings, and returned arrays must be released with
  the specific matching free/destroy function documented on each API
- some helpers return pointers to statically owned strings; those must not be
  freed

For the C++ API:

- wrapper classes are move-only and use RAII
- copied result objects are returned as standard C++ value types
- callers normally do not need to manage the underlying C pointers directly

## Typical workflow

For both APIs, the usual flow is:

1. create and fill a config object
2. create the engine or recognizer
3. create a stream if the feature is stream-based
4. feed audio or text
5. run decode/compute/generate
6. read back results
7. destroy resources, or let the C++ wrappers clean them up automatically

## Recommended entry points

Start with:

- [`c-api.h`](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/c-api/c-api.h)
  for the plain C API
- [`cxx-api.h`](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/c-api/cxx-api.h)
  for the C++ wrapper

Representative example programs live in:

- [`c-api-examples/`](https://github.com/k2-fsa/sherpa-onnx/tree/master/c-api-examples)
- [`cxx-api-examples/`](https://github.com/k2-fsa/sherpa-onnx/tree/master/cxx-api-examples)

Useful examples include:

- [`decode-file-c-api.c`](https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/decode-file-c-api.c)
- [`whisper-c-api.c`](https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/whisper-c-api.c)
- [`cohere-transcribe-c-api.c`](https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/cohere-transcribe-c-api.c)
- [`sense-voice-c-api.c`](https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/sense-voice-c-api.c)
- [`nemo-parakeet-c-api.c`](https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/nemo-parakeet-c-api.c)
- [`streaming-zipformer-with-hr-cxx-api.cc`](https://github.com/k2-fsa/sherpa-onnx/blob/master/cxx-api-examples/streaming-zipformer-with-hr-cxx-api.cc)
- [`sense-voice-cxx-api.cc`](https://github.com/k2-fsa/sherpa-onnx/blob/master/cxx-api-examples/sense-voice-cxx-api.cc)
- [`pocket-tts-en-cxx-api.cc`](https://github.com/k2-fsa/sherpa-onnx/blob/master/cxx-api-examples/pocket-tts-en-cxx-api.cc)
- [`vad-cxx-api.cc`](https://github.com/k2-fsa/sherpa-onnx/blob/master/cxx-api-examples/vad-cxx-api.cc)

## Generating the documentation

From `sherpa-onnx/c-api/`, run:

```bash
doxygen Doxyfile
```

HTML output is written to:

```text
doxygen-docs/html/
```
