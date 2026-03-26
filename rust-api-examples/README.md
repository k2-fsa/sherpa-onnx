# Introduction

This folder contains examples that use the `sherpa-onnx` Rust crate maintained in
this repository.

## Setup

For most users, you don't need to configure Rust linking details manually.

Just enter this directory and run one of the helper scripts below. Each script
downloads the required model files automatically if needed.

For example:

```bash
./run-version.sh
```

You can also run examples directly with Cargo:

```bash
cargo run --example version
```

The default Rust setup uses **static** linking.

The first build may download the matching sherpa-onnx native libraries for your
platform automatically. This process is usually automatic and mostly invisible
to the user.

If you want **shared** libraries instead of the default static behavior, use:

```bash
cargo run --no-default-features --features shared --example version
```

If you want to customize which libraries are used, set `SHERPA_ONNX_LIB_DIR`,
choose shared instead of the default behavior, or configure the crate directly
in your own Cargo project, see
[for-advanced-users.md](./for-advanced-users.md).

## Examples

| # | Example | Description |
|---|---------|-------------|
| 1 | [version](#example-1-show-sherpa-onnx-version) | Show the sherpa-onnx version |
| 2 | [pocket_tts](#example-2-tts-with-pocket-tts-zero-shot-voice-cloning) | Text-to-speech with zero-shot voice cloning using a reference audio |
| 3 | [supertonic_tts](#example-3-tts-with-supertonic-tts) | Text-to-speech with Supertonic TTS (multi-speaker, multi-language) |
| 4 | [zipvoice_tts](#example-4-tts-with-zipvoice-zero-shot-voice-cloning) | Text-to-speech with ZipVoice zero-shot voice cloning |
| 5 | [vits_tts](#example-5-tts-with-vits-english-piper) | Text-to-speech with a standalone VITS Piper model (English) |
| 6 | [vits_tts](#example-6-tts-with-vits-german-piper) | Text-to-speech with a standalone VITS Piper model (German) |
| 7 | [matcha_tts_en](#example-7-tts-with-matcha-english) | Text-to-speech with Matcha TTS (English) |
| 8 | [matcha_tts_zh](#example-8-tts-with-matcha-chinese) | Text-to-speech with Matcha TTS (Chinese) |
| 9 | [kokoro_tts_en](#example-9-tts-with-kokoro-english) | Text-to-speech with Kokoro TTS (English) |
| 10 | [kokoro_tts_zh_en](#example-10-tts-with-kokoro-chinese--english) | Text-to-speech with Kokoro TTS (Chinese + English) |
| 11 | [kitten_tts_en](#example-11-tts-with-kitten-english) | Text-to-speech with Kitten TTS (English) |
| 12 | [streaming_zipformer_en](#example-12-asr-with-streaming-zipformer-english) | Streaming ASR with zipformer transducer (English) |
| 13 | [streaming_zipformer_zh_en](#example-13-asr-with-streaming-zipformer-chinese--english) | Streaming ASR with zipformer transducer (Chinese + English) |
| 14 | [streaming_zipformer_microphone](#example-14-asr-with-streaming-zipformer-with-a-microphone-real-time-asr) | Real-time streaming ASR from microphone input |
| 15 | [zipformer_en](#example-15-asr-with-non-streaming-zipformer-english) | Non-streaming ASR with zipformer transducer (English) |
| 16 | [zipformer_zh_en](#example-16-asr-with-non-streaming-zipformer-chinese--english) | Non-streaming ASR with zipformer transducer (Chinese + English) |
| 17 | [zipformer_vi](#example-17-asr-with-non-streaming-zipformer-vietnamese) | Non-streaming ASR with zipformer transducer (Vietnamese) |
| 18 | [nemo_parakeet](#example-18-asr-with-non-streaming-nemo-parakeet-english) | Non-streaming ASR with Nemo Parakeet TDT transducer (English) |
| 19 | [fire_red_asr_ctc](#example-19-asr-with-non-streaming-fireredasr-ctc-chinese--english) | Non-streaming ASR with FireRedASR CTC model (Chinese + English) |
| 20 | [moonshine_v2](#example-20-asr-with-non-streaming-moonshine-v2-english) | Non-streaming ASR with Moonshine v2 (English) |
| 21 | [sense_voice](#example-21-asr-with-non-streaming-sensevoice) | Non-streaming ASR with SenseVoice (Chinese, English, Japanese, Korean, Cantonese) |
| 22 | [qwen3_asr](#example-22-asr-with-non-streaming-qwen3-asr) | Non-streaming ASR with Qwen3 ASR (multilingual) |
| 23 | [silero_vad_remove_silence](#example-23-remove-silences-from-a-file-using-silerovad) | Remove silences from an audio file using Silero VAD |
| 23 | [offline_speech_enhancement_gtcrn](#example-23-offline-speech-enhancement-with-gtcrn) | Offline speech enhancement with GTCRN |
| 24 | [offline_speech_enhancement_dpdfnet](#example-24-offline-speech-enhancement-with-dpdfnet) | Offline speech enhancement with DPDFNet |
| 25 | [streaming_speech_enhancement_gtcrn](#example-25-streaming-speech-enhancement-with-gtcrn) | Streaming speech enhancement with GTCRN |
| 26 | [streaming_speech_enhancement_dpdfnet](#example-26-streaming-speech-enhancement-with-dpdfnet) | Streaming speech enhancement with DPDFNet |
| 27 | [online_punctuation](#example-27-online-punctuation) | Add punctuation to text using online punctuation model |
| 28 | [keyword_spotter](#example-28-keyword-spotter) | Detect keywords from audio using a Zipformer KWS model |
| 29 | [spoken_language_identification](#example-29-spoken-language-identification) | Detect the spoken language in a wave file using Whisper |
| 30 | [offline_punctuation](#example-30-offline-punctuation) | Add punctuation to text using an offline punctuation model |
| 31 | [audio_tagging_zipformer](#example-31-audio-tagging-with-a-zipformer-model) | Audio tagging with a Zipformer model |
| 32 | [audio_tagging_ced](#example-32-audio-tagging-with-a-ced-model) | Audio tagging with a CED model |
| 33 | [speaker_embedding_extractor](#example-33-speaker-embedding-extractor) | Compute a speaker embedding from a wave file |
| 34 | [speaker_embedding_manager](#example-34-speaker-embedding-manager) | Register, search, verify, and remove speakers using embeddings |
| 35 | [speaker_embedding_cosine_similarity](#example-35-speaker-embedding-cosine-similarity) | Compute cosine similarity from three speaker embeddings |
| 36 | [offline_speaker_diarization](#example-36-offline-speaker-diarization) | Offline speaker diarization with pyannote segmentation and 3D-Speaker embeddings |
| 37 | [sense_voice_simulate_streaming_microphone](#example-37-simulated-streaming-asr-with-sensevoice-and-vad-from-microphone) | Simulated streaming ASR with SenseVoice and VAD from microphone |
| 38 | [fire_red_asr_ctc_simulate_streaming_microphone](#example-38-simulated-streaming-asr-with-fireredasr-ctc-and-vad-from-microphone) | Simulated streaming ASR with FireRedASR CTC and VAD from microphone |
| 39 | [parakeet_tdt_ctc_simulate_streaming_microphone](#example-39-simulated-streaming-asr-with-parakeet-tdt-ctc-and-vad-from-microphone) | Simulated streaming ASR with Parakeet TDT CTC and VAD from microphone |
| 40 | [parakeet_tdt_simulate_streaming_microphone](#example-40-simulated-streaming-asr-with-parakeet-tdt-transducer-and-vad-from-microphone) | Simulated streaming ASR with Parakeet TDT transducer and VAD from microphone |
| 41 | [wenet_ctc_simulate_streaming_microphone](#example-41-simulated-streaming-asr-with-wenet-ctc-and-vad-from-microphone) | Simulated streaming ASR with WeNet CTC and VAD from microphone |
| 42 | [zipformer_ctc_simulate_streaming_microphone](#example-42-simulated-streaming-asr-with-zipformer-ctc-and-vad-from-microphone) | Simulated streaming ASR with Zipformer CTC and VAD from microphone |
| 43 | [zipformer_transducer_simulate_streaming_microphone](#example-43-simulated-streaming-asr-with-zipformer-transducer-and-vad-from-microphone) | Simulated streaming ASR with Zipformer transducer and VAD from microphone |
| 44 | [zipformer_transducer_simulate_streaming_microphone](#example-44-simulated-streaming-asr-with-zipformer-transducer-japanese-and-vad-from-microphone) | Simulated streaming ASR with Zipformer transducer (Japanese) and VAD from microphone |
| 45 | [qwen3_asr_simulate_streaming_microphone](#example-45-simulated-streaming-asr-with-qwen3-asr-and-vad-from-microphone) | Simulated streaming ASR with Qwen3 ASR and VAD from microphone |

## Run it

Each helper script downloads the required files if needed.

### Example 1: Show sherpa-onnx version

```bash
./run-version.sh
```

For macOS, you can run
```
otool -l target/debug/examples/version | grep -A2 LC_RPATH
```
to check the RPATH for shared builds.

### Example 2: TTS with Pocket TTS (zero-shot voice cloning)

```bash
./run-pocket-tts.sh
```

### Example 3: TTS with Supertonic TTS

```bash
./run-supertonic-tts.sh
```

### Example 4: TTS with ZipVoice zero-shot voice cloning

```bash
./run-zipvoice-tts.sh
```


### Example 5: TTS with VITS (English Piper)

```bash
./run-vits-en.sh
```

### Example 6: TTS with VITS (German Piper)

```bash
./run-vits-de.sh
```

### Example 7: TTS with Matcha (English)

```bash
./run-matcha-tts-en.sh
```

### Example 8: TTS with Matcha (Chinese)

```bash
./run-matcha-tts-zh.sh
```

### Example 9: TTS with Kokoro (English)

```bash
./run-kokoro-tts-en.sh
```

### Example 10: TTS with Kokoro (Chinese + English)

```bash
./run-kokoro-tts-zh-en.sh
```

### Example 11: TTS with Kitten (English)

```bash
./run-kitten-tts-en.sh
```

### Example 12: ASR with streaming zipformer (English)

```bash
./run-streaming-zipformer-en.sh
```

### Example 13: ASR with streaming zipformer (Chinese + English)

```bash
./run-streaming-zipformer-zh-en.sh
```

### Example 14: ASR with streaming zipformer (with a microphone, real-time ASR)

```bash
./run-streaming-zipformer-microphone-zh-en.sh
```

### Example 15: ASR with non-streaming zipformer (English)

```bash
./run-zipformer-en.sh
```

### Example 16: ASR with non-streaming zipformer (Chinese + English)

```bash
./run-zipformer-zh-en.sh
```

### Example 17: ASR with non-streaming zipformer (Vietnamese)

```bash
./run-zipformer-vi.sh
```

### Example 18: ASR with non-streaming Nemo Parakeet (English)

```bash
./run-nemo-parakeet-en.sh
```

### Example 19: ASR with non-streaming FireRedASR CTC (Chinese + English)

```bash
./run-fire-red-asr-ctc.sh
```

### Example 20: ASR with non-streaming Moonshine v2 (English)

```bash
./run-moonshine-v2.sh
```

### Example 21: ASR with non-streaming SenseVoice

```bash
./run-sense-voice.sh
```

### Example 22: ASR with non-streaming Qwen3 ASR

```bash
./run-qwen3-asr.sh
```

### Example 23: Remove silences from a file using SileroVAD

```bash
./run-silero-vad-remove-silence.sh
```

### Example 24: Offline speech enhancement with GTCRN

```bash
./run-offline-speech-enhancement-gtcrn.sh
```

### Example 25: Offline speech enhancement with DPDFNet

```bash
./run-offline-speech-enhancement-dpdfnet.sh
```

### Example 26: Streaming speech enhancement with GTCRN

```bash
./run-streaming-speech-enhancement-gtcrn.sh
```

### Example 27: Streaming speech enhancement with DPDFNet

```bash
./run-streaming-speech-enhancement-dpdfnet.sh
```

### Example 28: Online punctuation

```bash
./run-online-punctuation.sh
```

### Example 29: Keyword spotter

```bash
./run-keyword-spotter.sh
```

### Example 30: Spoken language identification

```bash
./run-spoken-language-identification.sh
```

### Example 31: Offline punctuation

```bash
./run-offline-punctuation.sh
```

### Example 32: Audio tagging with a Zipformer model

```bash
./run-audio-tagging-zipformer.sh
```

### Example 33: Audio tagging with a CED model

```bash
./run-audio-tagging-ced.sh
```


### Example 34: Speaker embedding extractor

```bash
./run-speaker-embedding-extractor.sh
```

### Example 35: Speaker embedding manager

```bash
./run-speaker-embedding-manager.sh
```


### Example 36: Speaker embedding cosine similarity

```bash
./run-speaker-embedding-cosine-similarity.sh
```


### Example 37: Offline speaker diarization

```bash
./run-offline-speaker-diarization.sh
```

### Example 38: Simulated streaming ASR with SenseVoice and VAD from microphone

This example uses Silero VAD to detect speech segments and runs the offline
SenseVoice recognizer on each detected segment, providing an experience
similar to streaming ASR.

```bash
./run-sense-voice-simulate-streaming-microphone.sh
```

### Example 39: Simulated streaming ASR with FireRedASR CTC and VAD from microphone

This example uses Silero VAD to detect speech segments and runs the offline
FireRedASR CTC recognizer on each detected segment.

```bash
./run-fire-red-asr-ctc-simulate-streaming-microphone.sh
```

### Example 40: Simulated streaming ASR with Parakeet TDT CTC and VAD from microphone

This example uses Silero VAD to detect speech segments and runs the offline
Parakeet TDT CTC recognizer on each detected segment (Japanese).

```bash
./run-parakeet-tdt-ctc-simulate-streaming-microphone.sh
```

### Example 41: Simulated streaming ASR with Parakeet TDT transducer and VAD from microphone

This example uses Silero VAD to detect speech segments and runs the offline
Parakeet TDT transducer recognizer on each detected segment (English).

```bash
./run-parakeet-tdt-simulate-streaming-microphone.sh
```

### Example 42: Simulated streaming ASR with WeNet CTC and VAD from microphone

This example uses Silero VAD to detect speech segments and runs the offline
WeNet CTC recognizer on each detected segment (Cantonese).

```bash
./run-wenet-ctc-simulate-streaming-microphone.sh
```

### Example 43: Simulated streaming ASR with Zipformer CTC and VAD from microphone

This example uses Silero VAD to detect speech segments and runs the offline
Zipformer CTC recognizer on each detected segment (Chinese).

```bash
./run-zipformer-ctc-simulate-streaming-microphone.sh
```

### Example 44: Simulated streaming ASR with Zipformer transducer and VAD from microphone

This example uses Silero VAD to detect speech segments and runs the offline
Zipformer transducer recognizer on each detected segment (Chinese).

```bash
./run-zipformer-transducer-simulate-streaming-microphone.sh
```

### Example 45: Simulated streaming ASR with Zipformer transducer (Japanese) and VAD from microphone

This example uses Silero VAD to detect speech segments and runs the offline
Zipformer transducer recognizer on each detected segment (Japanese,
reazonspeech model).

```bash
./run-zipformer-ja-reazonspeech-simulate-streaming-microphone.sh
```

### Example 45: Simulated streaming ASR with Qwen3 ASR and VAD from microphone

This example uses Silero VAD to detect speech segments and runs the offline
Qwen3 ASR recognizer on each detected segment.

```bash
./run-qwen3-asr-simulate-streaming-microphone.sh
```
