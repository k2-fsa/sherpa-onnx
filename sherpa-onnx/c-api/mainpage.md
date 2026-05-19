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

## Model-specific documentation

Each model family has its own documentation page with config examples:

- @ref offline_asr — Non-streaming ASR: Zipformer Transducer, Zipformer CTC, Whisper, SenseVoice, Paraformer, Moonshine, FireRedAsr, Dolphin, Canary, Cohere, WeNet, Omnilingual, FunASR Nano, Qwen3, MedASR, TeleSpeech, GigaAM v2, Parakeet TDT, NeMo CTC
- @ref online_asr — Streaming ASR: Transducer (Zipformer), Nemotron, Paraformer, Zipformer2 CTC, T-One CTC
- @ref tts — Text-to-Speech: Kokoro, VITS (Piper), Matcha, Kitten, ZipVoice, Pocket, Supertonic
- @ref vad — Voice Activity Detection: Silero VAD, Ten VAD
- @ref audio_tagging — Audio Tagging: Zipformer, CED
- @ref punctuation — Punctuation: Offline (CT-Transformer), Online (CNN-BiLSTM)
- @ref speech_enhancement — Speech Enhancement: GTCRN, DPDFNet (offline and online)
- @ref source_separation — Source Separation: Spleeter, UVR
- @ref speaker_diarization — Speaker Diarization: Pyannote segmentation + embedding clustering
- @ref speaker_embedding — Speaker Embedding: extraction, enrollment, search, verification
- @ref spoken_language_id — Spoken Language Identification: Whisper-based
- @ref keyword_spotting — Keyword Spotting: Zipformer KWS
- @ref resampler — Linear Resampler

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

**Offline ASR (C API):**
- [`whisper-c-api.c`](https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/whisper-c-api.c)
- [`sense-voice-c-api.c`](https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/sense-voice-c-api.c)
- [`paraformer-c-api.c`](https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/paraformer-c-api.c)
- [`moonshine-c-api.c`](https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/moonshine-c-api.c)
- [`zipformer-c-api.c`](https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/zipformer-c-api.c)
- [`nemo-parakeet-c-api.c`](https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/nemo-parakeet-c-api.c)
- [`nemo-giga-am-v2-c-api.c`](https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/nemo-giga-am-v2-c-api.c)
- [`nemo-ctc-c-api.c`](https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/nemo-ctc-c-api.c)
- [`cohere-transcribe-c-api.c`](https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/cohere-transcribe-c-api.c)
- [`dolphin-ctc-c-api.c`](https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/dolphin-ctc-c-api.c)

**Streaming ASR (C API):**
- [`streaming-zipformer-c-api.c`](https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/streaming-zipformer-c-api.c)
- [`streaming-nemotron-c-api.c`](https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/streaming-nemotron-c-api.c)
- [`streaming-paraformer-c-api.c`](https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/streaming-paraformer-c-api.c)

**TTS (C API):**
- [`kokoro-tts-en-c-api.c`](https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/kokoro-tts-en-c-api.c)
- [`pocket-tts-en-c-api.c`](https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/pocket-tts-en-c-api.c)
- [`offline-tts-c-api.c`](https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/offline-tts-c-api.c)

**Other features (C API):**
- [`vad-whisper-c-api.c`](https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/vad-whisper-c-api.c)
- [`audio-tagging-c-api.c`](https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/audio-tagging-c-api.c)
- [`speaker-identification-c-api.c`](https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/speaker-identification-c-api.c)
- [`offline-speaker-diarization-c-api.c`](https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/offline-speaker-diarization-c-api.c)
- [`spoken-language-identification-c-api.c`](https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/spoken-language-identification-c-api.c)
- [`kws-c-api.c`](https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/kws-c-api.c)

**C++ API examples:**
- [`streaming-zipformer-cxx-api.cc`](https://github.com/k2-fsa/sherpa-onnx/blob/master/cxx-api-examples/streaming-zipformer-cxx-api.cc)
- [`streaming-nemotron-cxx-api.cc`](https://github.com/k2-fsa/sherpa-onnx/blob/master/cxx-api-examples/streaming-nemotron-cxx-api.cc)
- [`paraformer-cxx-api.cc`](https://github.com/k2-fsa/sherpa-onnx/blob/master/cxx-api-examples/paraformer-cxx-api.cc)
- [`nemo-ctc-cxx-api.cc`](https://github.com/k2-fsa/sherpa-onnx/blob/master/cxx-api-examples/nemo-ctc-cxx-api.cc)
- [`offline-tts-piper-cxx-api.cc`](https://github.com/k2-fsa/sherpa-onnx/blob/master/cxx-api-examples/offline-tts-piper-cxx-api.cc)
- [`pocket-tts-en-cxx-api.cc`](https://github.com/k2-fsa/sherpa-onnx/blob/master/cxx-api-examples/pocket-tts-en-cxx-api.cc)
- [`vad-cxx-api.cc`](https://github.com/k2-fsa/sherpa-onnx/blob/master/cxx-api-examples/vad-cxx-api.cc)
- [`offline-speaker-diarization-cxx-api.cc`](https://github.com/k2-fsa/sherpa-onnx/blob/master/cxx-api-examples/offline-speaker-diarization-cxx-api.cc)
- [`speaker-identification-cxx-api.cc`](https://github.com/k2-fsa/sherpa-onnx/blob/master/cxx-api-examples/speaker-identification-cxx-api.cc)
- [`spoken-language-identification-cxx-api.cc`](https://github.com/k2-fsa/sherpa-onnx/blob/master/cxx-api-examples/spoken-language-identification-cxx-api.cc)

## Generating the documentation

From `sherpa-onnx/c-api/`, run:

```bash
doxygen Doxyfile
```

HTML output is written to:

```text
doxygen-docs/html/
```
