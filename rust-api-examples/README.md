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
| 4 | [zipvoice_tts_lexicon](#example-4-tts-with-zipvoice-zero-shot-voice-cloning) | Text-to-speech with ZipVoice zero-shot voice cloning (lexicon) |
| 5 | [vits_tts_lexicon](#example-5-tts-with-vits-english-piper-lexicon) | Text-to-speech with a standalone VITS Piper model (English, lexicon) |
| 6 | [vits_tts_phonemize](#example-6-tts-with-vits-english-piper-piper-phonemize) | Text-to-speech with a standalone VITS Piper model (English, piper-phonemize) |
| 7 | [vits_tts_de_phonemize](#example-7-tts-with-vits-german-piper-piper-phonemize) | Text-to-speech with a standalone VITS Piper model (German, piper-phonemize — no German lexicon is published yet) |
| 8 | [matcha_tts_en_lexicon](#example-8-tts-with-matcha-english-lexicon) | Text-to-speech with Matcha TTS (English, lexicon) |
| 9 | [matcha_tts_en_phonemize](#example-9-tts-with-matcha-english-piper-phonemize) | Text-to-speech with Matcha TTS (English, piper-phonemize) |
| 10 | [matcha_tts_zh_lexicon](#example-10-tts-with-matcha-chinese) | Text-to-speech with Matcha TTS (Chinese, lexicon) |
| 11 | [kokoro_tts_en_lexicon](#example-11-tts-with-kokoro-english-lexicon) | Text-to-speech with Kokoro TTS (English, lexicon) |
| 12 | [kokoro_tts_en_phonemize](#example-12-tts-with-kokoro-english-piper-phonemize) | Text-to-speech with Kokoro TTS (English, piper-phonemize) |
| 13 | [kokoro_tts_zh_en_lexicon](#example-13-tts-with-kokoro-chinese--english) | Text-to-speech with Kokoro TTS (Chinese + English, lexicon) |
| 14 | [kitten_tts_en_lexicon](#example-14-tts-with-kitten-english-lexicon) | Text-to-speech with Kitten TTS (English, lexicon) |
| 15 | [kitten_tts_en_phonemize](#example-15-tts-with-kitten-english-piper-phonemize) | Text-to-speech with Kitten TTS (English, piper-phonemize) |
| 16 | [streaming_zipformer_en](#example-16-asr-with-streaming-zipformer-english) | Streaming ASR with zipformer transducer (English) |
| 17 | [streaming_zipformer_zh_en](#example-17-asr-with-streaming-zipformer-chinese--english) | Streaming ASR with zipformer transducer (Chinese + English) |
| 18 | [streaming_zipformer_microphone](#example-18-asr-with-streaming-zipformer-with-a-microphone-real-time-asr) | Real-time streaming ASR from microphone input |
| 19 | [zipformer_en](#example-19-asr-with-non-streaming-zipformer-english) | Non-streaming ASR with zipformer transducer (English) |
| 20 | [zipformer_zh_en](#example-20-asr-with-non-streaming-zipformer-chinese--english) | Non-streaming ASR with zipformer transducer (Chinese + English) |
| 21 | [zipformer_vi](#example-21-asr-with-non-streaming-zipformer-vietnamese) | Non-streaming ASR with zipformer transducer (Vietnamese) |
| 22 | [nemo_parakeet](#example-22-asr-with-non-streaming-nemo-parakeet-english) | Non-streaming ASR with Nemo Parakeet TDT transducer (English) |
| 23 | [fire_red_asr_ctc](#example-23-asr-with-non-streaming-fireredasr-ctc-chinese--english) | Non-streaming ASR with FireRedASR CTC model (Chinese + English) |
| 24 | [moonshine_v2](#example-24-asr-with-non-streaming-moonshine-v2-english) | Non-streaming ASR with Moonshine v2 (English) |
| 25 | [sense_voice](#example-25-asr-with-non-streaming-sensevoice) | Non-streaming ASR with SenseVoice (Chinese, English, Japanese, Korean, Cantonese) |
| 26 | [qwen3_asr](#example-26-asr-with-non-streaming-qwen3-asr) | Non-streaming ASR with Qwen3 ASR (multilingual) |
| 27 | [cohere_transcribe](#example-27-asr-with-non-streaming-cohere-transcribe) | Non-streaming ASR with Cohere Transcribe (multilingual) |
| 28 | [silero_vad_remove_silence](#example-28-remove-silences-from-a-file-using-silerovad) | Remove silences from an audio file using Silero VAD |
| 29 | [offline_speech_enhancement_gtcrn](#example-29-offline-speech-enhancement-with-gtcrn) | Offline speech enhancement with GTCRN |
| 30 | [offline_speech_enhancement_dpdfnet](#example-30-offline-speech-enhancement-with-dpdfnet) | Offline speech enhancement with DPDFNet |
| 31 | [streaming_speech_enhancement_gtcrn](#example-31-streaming-speech-enhancement-with-gtcrn) | Streaming speech enhancement with GTCRN |
| 32 | [streaming_speech_enhancement_dpdfnet](#example-32-streaming-speech-enhancement-with-dpdfnet) | Streaming speech enhancement with DPDFNet |
| 33 | [online_punctuation](#example-33-online-punctuation) | Add punctuation to text using online punctuation model |
| 34 | [keyword_spotter](#example-34-keyword-spotter) | Detect keywords from audio using a Zipformer KWS model |
| 35 | [spoken_language_identification](#example-35-spoken-language-identification) | Detect the spoken language in a wave file using Whisper |
| 36 | [offline_punctuation](#example-36-offline-punctuation) | Add punctuation to text using an offline punctuation model |
| 37 | [audio_tagging_zipformer](#example-37-audio-tagging-with-a-zipformer-model) | Audio tagging with a Zipformer model |
| 38 | [audio_tagging_ced](#example-38-audio-tagging-with-a-ced-model) | Audio tagging with a CED model |
| 39 | [speaker_embedding_extractor](#example-39-speaker-embedding-extractor) | Compute a speaker embedding from a wave file |
| 40 | [speaker_embedding_manager](#example-40-speaker-embedding-manager) | Register, search, verify, and remove speakers using embeddings |
| 41 | [speaker_embedding_cosine_similarity](#example-41-speaker-embedding-cosine-similarity) | Compute cosine similarity from three speaker embeddings |
| 42 | [offline_speaker_diarization](#example-42-offline-speaker-diarization) | Offline speaker diarization with pyannote segmentation and 3D-Speaker embeddings |
| 43 | [sense_voice_simulate_streaming_microphone](#example-43-simulated-streaming-asr-with-sensevoice-and-vad-from-microphone) | Simulated streaming ASR with SenseVoice and VAD from microphone |
| 44 | [fire_red_asr_ctc_simulate_streaming_microphone](#example-44-simulated-streaming-asr-with-fireredasr-ctc-and-vad-from-microphone) | Simulated streaming ASR with FireRedASR CTC and VAD from microphone |
| 45 | [parakeet_tdt_ctc_simulate_streaming_microphone](#example-45-simulated-streaming-asr-with-parakeet-tdt-ctc-and-vad-from-microphone) | Simulated streaming ASR with Parakeet TDT CTC and VAD from microphone |
| 46 | [parakeet_tdt_simulate_streaming_microphone](#example-46-simulated-streaming-asr-with-parakeet-tdt-transducer-and-vad-from-microphone) | Simulated streaming ASR with Parakeet TDT transducer and VAD from microphone |
| 47 | [wenet_ctc_simulate_streaming_microphone](#example-47-simulated-streaming-asr-with-wenet-ctc-and-vad-from-microphone) | Simulated streaming ASR with WeNet CTC and VAD from microphone |
| 48 | [zipformer_ctc_simulate_streaming_microphone](#example-48-simulated-streaming-asr-with-zipformer-ctc-and-vad-from-microphone) | Simulated streaming ASR with Zipformer CTC and VAD from microphone |
| 49 | [zipformer_transducer_simulate_streaming_microphone](#example-49-simulated-streaming-asr-with-zipformer-transducer-and-vad-from-microphone) | Simulated streaming ASR with Zipformer transducer and VAD from microphone |
| 50 | [zipformer_transducer_simulate_streaming_microphone](#example-50-simulated-streaming-asr-with-zipformer-transducer-japanese-and-vad-from-microphone) | Simulated streaming ASR with Zipformer transducer (Japanese) and VAD from microphone |
| 51 | [qwen3_asr_simulate_streaming_microphone](#example-51-simulated-streaming-asr-with-qwen3-asr-and-vad-from-microphone) | Simulated streaming ASR with Qwen3 ASR and VAD from microphone |
| 52 | [whisper](#example-52-asr-with-non-streaming-whisper) | Non-streaming ASR with Whisper (multilingual) |
| 55 | [paraformer](#example-55-asr-with-non-streaming-paraformer) | Non-streaming ASR with Paraformer |

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
./run-zipvoice-tts-lexicon.sh
```


### Example 5: TTS with VITS (English Piper, lexicon)

```bash
./run-vits-en-lexicon.sh
```

### Example 6: TTS with VITS (English Piper, piper-phonemize)

The text is phonemized outside of sherpa-onnx with the piper-phonemize crate.

```bash
./run-vits-en-phonemize.sh
```

### Example 7: TTS with VITS (German Piper, piper-phonemize)

There is no published German lexicon yet, so this example uses the
piper-phonemize frontend.

```bash
./run-vits-de-phonemize.sh
```

### Example 8: TTS with Matcha (English, lexicon)

```bash
./run-matcha-tts-en-lexicon.sh
```

### Example 9: TTS with Matcha (English, piper-phonemize)

```bash
./run-matcha-tts-en-phonemize.sh
```

### Example 10: TTS with Matcha (Chinese)

```bash
./run-matcha-tts-zh-lexicon.sh
```

### Example 11: TTS with Kokoro (English, lexicon)

```bash
./run-kokoro-tts-en-lexicon.sh
```

### Example 12: TTS with Kokoro (English, piper-phonemize)

```bash
./run-kokoro-tts-en-phonemize.sh
```

### Example 13: TTS with Kokoro (Chinese + English)

```bash
./run-kokoro-tts-zh-en-lexicon.sh
```

### Example 14: TTS with Kitten (English, lexicon)

```bash
./run-kitten-tts-en-lexicon.sh
```

### Example 15: TTS with Kitten (English, piper-phonemize)

```bash
./run-kitten-tts-en-phonemize.sh
```

### Example 16: ASR with streaming zipformer (English)

```bash
./run-streaming-zipformer-en.sh
```

### Example 17: ASR with streaming zipformer (Chinese + English)

```bash
./run-streaming-zipformer-zh-en.sh
```

### Example 18: ASR with streaming zipformer (with a microphone, real-time ASR)

```bash
./run-streaming-zipformer-microphone-zh-en.sh
```

### Example 19: ASR with non-streaming zipformer (English)

```bash
./run-zipformer-en.sh
```

### Example 20: ASR with non-streaming zipformer (Chinese + English)

```bash
./run-zipformer-zh-en.sh
```

### Example 21: ASR with non-streaming zipformer (Vietnamese)

```bash
./run-zipformer-vi.sh
```

### Example 22: ASR with non-streaming Nemo Parakeet (English)

```bash
./run-nemo-parakeet-en.sh
```

### Example 23: ASR with non-streaming FireRedASR CTC (Chinese + English)

```bash
./run-fire-red-asr-ctc.sh
```

### Example 24: ASR with non-streaming Moonshine v2 (English)

```bash
./run-moonshine-v2.sh
```

### Example 25: ASR with non-streaming SenseVoice

```bash
./run-sense-voice.sh
```

### Example 26: ASR with non-streaming Qwen3 ASR

```bash
./run-qwen3-asr.sh
```

### Example 27: ASR with non-streaming Cohere Transcribe

```bash
./run-cohere-transcribe.sh
```

### Example 28: Remove silences from a file using SileroVAD

```bash
./run-silero-vad-remove-silence.sh
```

### Example 29: Offline speech enhancement with GTCRN

```bash
./run-offline-speech-enhancement-gtcrn.sh
```

### Example 30: Offline speech enhancement with DPDFNet

```bash
./run-offline-speech-enhancement-dpdfnet.sh
```

### Example 31: Streaming speech enhancement with GTCRN

```bash
./run-streaming-speech-enhancement-gtcrn.sh
```

### Example 32: Streaming speech enhancement with DPDFNet

```bash
./run-streaming-speech-enhancement-dpdfnet.sh
```

### Example 33: Online punctuation

```bash
./run-online-punctuation.sh
```

### Example 34: Keyword spotter

```bash
./run-keyword-spotter.sh
```

### Example 35: Spoken language identification

```bash
./run-spoken-language-identification.sh
```

### Example 36: Offline punctuation

```bash
./run-offline-punctuation.sh
```

### Example 37: Audio tagging with a Zipformer model

```bash
./run-audio-tagging-zipformer.sh
```

### Example 38: Audio tagging with a CED model

```bash
./run-audio-tagging-ced.sh
```


### Example 39: Speaker embedding extractor

```bash
./run-speaker-embedding-extractor.sh
```

### Example 40: Speaker embedding manager

```bash
./run-speaker-embedding-manager.sh
```


### Example 41: Speaker embedding cosine similarity

```bash
./run-speaker-embedding-cosine-similarity.sh
```


### Example 42: Offline speaker diarization

```bash
./run-offline-speaker-diarization.sh
```

### Example 43: Simulated streaming ASR with SenseVoice and VAD from microphone

This example uses Silero VAD to detect speech segments and runs the offline
SenseVoice recognizer on each detected segment, providing an experience
similar to streaming ASR.

```bash
./run-sense-voice-simulate-streaming-microphone.sh
```

### Example 44: Simulated streaming ASR with FireRedASR CTC and VAD from microphone

This example uses Silero VAD to detect speech segments and runs the offline
FireRedASR CTC recognizer on each detected segment.

```bash
./run-fire-red-asr-ctc-simulate-streaming-microphone.sh
```

### Example 45: Simulated streaming ASR with Parakeet TDT CTC and VAD from microphone

This example uses Silero VAD to detect speech segments and runs the offline
Parakeet TDT CTC recognizer on each detected segment (Japanese).

```bash
./run-parakeet-tdt-ctc-simulate-streaming-microphone.sh
```

### Example 46: Simulated streaming ASR with Parakeet TDT transducer and VAD from microphone

This example uses Silero VAD to detect speech segments and runs the offline
Parakeet TDT transducer recognizer on each detected segment (English).

```bash
./run-parakeet-tdt-simulate-streaming-microphone.sh
```

### Example 47: Simulated streaming ASR with WeNet CTC and VAD from microphone

This example uses Silero VAD to detect speech segments and runs the offline
WeNet CTC recognizer on each detected segment (Cantonese).

```bash
./run-wenet-ctc-simulate-streaming-microphone.sh
```

### Example 48: Simulated streaming ASR with Zipformer CTC and VAD from microphone

This example uses Silero VAD to detect speech segments and runs the offline
Zipformer CTC recognizer on each detected segment (Chinese).

```bash
./run-zipformer-ctc-simulate-streaming-microphone.sh
```

### Example 49: Simulated streaming ASR with Zipformer transducer and VAD from microphone

This example uses Silero VAD to detect speech segments and runs the offline
Zipformer transducer recognizer on each detected segment (Chinese).

```bash
./run-zipformer-transducer-simulate-streaming-microphone.sh
```

### Example 50: Simulated streaming ASR with Zipformer transducer (Japanese) and VAD from microphone

This example uses Silero VAD to detect speech segments and runs the offline
Zipformer transducer recognizer on each detected segment (Japanese,
reazonspeech model).

```bash
./run-zipformer-ja-reazonspeech-simulate-streaming-microphone.sh
```

### Example 51: Simulated streaming ASR with Qwen3 ASR and VAD from microphone

This example uses Silero VAD to detect speech segments and runs the offline
Qwen3 ASR recognizer on each detected segment.

```bash
./run-qwen3-asr-simulate-streaming-microphone.sh
```

### Example 52: ASR with non-streaming Whisper

```bash
./run-whisper.sh
```

### Example 53: ASR with non-streaming FunASR Nano

```bash
./run-funasr-nano.sh
```

### Example 54: Remove silences from a file using ten-vad

```bash
./run-ten-vad-remove-silence.sh
```

### Example 55: ASR with non-streaming Paraformer

```bash
./run-paraformer.sh