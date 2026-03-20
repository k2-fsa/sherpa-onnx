# Introduction

This folder uses Rust API maintained by us.

## Setup library path

### Method 1 (Build from source, support only shared libs right now)

```bash
export SHERPA_ONNX_LIB_DIR=/Users/fangjun/open-source/sherpa-onnx/build/install/lib
export RUSTFLAGS="-C link-arg=-Wl,-rpath,$SHERPA_ONNX_LIB_DIR"
```

### Method 2 (Download pre-built libs)

```bash
# You can choose any directory you like
cd $HOME/Downloads

# We use version v1.12.31 below as an example.
# Please always use the latest version from
# https://github.com/k2-fsa/sherpa-onnx/releases

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.12.31/sherpa-onnx-v1.12.31-osx-universal2-shared.tar.bz2
tar xvf sherpa-onnx-v1.12.31-osx-universal2-shared.tar.bz2
rm sherpa-onnx-v1.12.31-osx-universal2-shared.tar.bz2

export SHERPA_ONNX_LIB_DIR=$HOME/Downloads/sherpa-onnx-v1.12.31-osx-universal2-shared/lib
export RUSTFLAGS="-C link-arg=-Wl,-rpath,$SHERPA_ONNX_LIB_DIR"
```

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
| 22 | [silero_vad_remove_silence](#example-22-remove-silences-from-a-file-using-silerovad) | Remove silences from an audio file using Silero VAD |
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
to check the RPATH.

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

### Example 22: Remove silences from a file using SileroVAD

```bash
./run-silero-vad-remove-silence.sh
```

### Example 23: Offline speech enhancement with GTCRN

```bash
./run-offline-speech-enhancement-gtcrn.sh
```

### Example 24: Offline speech enhancement with DPDFNet

```bash
./run-offline-speech-enhancement-dpdfnet.sh
```

### Example 25: Streaming speech enhancement with GTCRN

```bash
./run-streaming-speech-enhancement-gtcrn.sh
```

### Example 26: Streaming speech enhancement with DPDFNet

```bash
./run-streaming-speech-enhancement-dpdfnet.sh
```

### Example 27: Online punctuation

```bash
./run-online-punctuation.sh
```

### Example 28: Keyword spotter

```bash
./run-keyword-spotter.sh
```

### Example 29: Spoken language identification

```bash
./run-spoken-language-identification.sh
```

### Example 30: Offline punctuation

```bash
./run-offline-punctuation.sh
```

### Example 31: Audio tagging with a Zipformer model

```bash
./run-audio-tagging-zipformer.sh
```

### Example 32: Audio tagging with a CED model

```bash
./run-audio-tagging-ced.sh
```


### Example 33: Speaker embedding extractor

```bash
./run-speaker-embedding-extractor.sh
```

### Example 34: Speaker embedding manager

```bash
./run-speaker-embedding-manager.sh
```


### Example 35: Speaker embedding cosine similarity

```bash
./run-speaker-embedding-cosine-similarity.sh
```


### Example 36: Offline speaker diarization

```bash
./run-offline-speaker-diarization.sh
```
