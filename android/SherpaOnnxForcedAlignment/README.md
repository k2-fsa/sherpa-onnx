# Forced alignment for Android

An Android app (Kotlin + Jetpack Compose + JNI) for
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) **forced alignment**:
given an audio file and its transcript, it computes the start/end time of
every word and lets you play, export and share the result.

Pre-built APKs (one per model and ABI) are in the
[apk-fa](https://github.com/k2-fsa/sherpa-onnx/releases/tag/apk-fa) GitHub
release; [../../scripts/mfa/android.md](../../scripts/mfa/android.md)
describes which APK to pick.

## What the app does

The UI follows [tauri-examples/doubao-asr-file](../../tauri-examples/doubao-asr-file);
it has two tabs.

### Align tab

- **Input file**
  - *Select audio or video file…* — wav, mp3, m4a/aac, flac, ogg, opus, wma,
    aiff, mp4, mov, webm, mkv, ... (anything FFmpeg can decode; the audio is
    decoded on-device to 16 kHz mono PCM).
  - *Example* — loads the model's bundled test wav (`en.wav`/`zh.wav`) and
    fills in its transcript, so the app can be tried without any external
    data.
  - *Record* — record from the microphone (16 kHz mono) and align the
    recording.
  - A preview player with a seek bar and a **Subtitles** overlay that shows
    the current word of the alignment while playing.
- **Transcript** — the transcript of the audio. Chinese text is tokenized per
  hanzi, English words are split on whitespace; punctuation is ignored. Words
  not in the model's lexicon are reported as out-of-vocabulary.
- **Align** — runs the forced aligner on-device (no audio is uploaded) and
  shows the result:
  - a **table of words** with their time stamps (Word | Time | Audio). Click a
    row (or ▶) to play just that word — it is cut sample-accurately from the
    decoded audio, so it never includes the neighbors; tap again to stop;
    💾 saves that word as a wav;
  - **Table / Timestamped / Plain** views and a search box with match
    navigation;
  - **Export** the result as a Praat **.TextGrid**, a timestamped **.txt** or
    a plain **.txt** (saved via the system file picker);
  - **Share** buttons that send the result to another app (WeChat, email,
    ...): the TextGrid as `<name>.TextGrid.txt` (the `.txt` suffix because
    receivers like WeChat reject a bare `.TextGrid`), and a `<name>.words.txt`
    with one `word start -- end` line per word;
  - **time statistics** right after aligning: audio duration, decode /
    align / total seconds and RTF;
  - **Play all** plays the words in sequence, skipping silence.
- **Log** — timestamped status lines.

### Info tab

The version of the app / sherpa-onnx / git revision / onnxruntime, the model
bundled in this APK (with download and documentation links), resources and
community links.

## How it works

- One forced-alignment model per APK, bundled in `app/src/main/assets/`
  (from the [fa-models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/fa-models)
  release; see [../../scripts/mfa/README.md](../../scripts/mfa/README.md)).
  `ModelConfig.kt` selects it via `SELECTED_MODEL_INDEX`.
- The aligner is the sherpa-onnx Kotlin API
  (`sherpa-onnx/kotlin-api/ForcedAligner.kt`, symlinked into
  `app/src/main/java/com/k2fsa/sherpa/onnx/`) over `libsherpa-onnx-jni.so`.
  The aligner is loaded in the background at app start and kept across tab
  switches, so the first run is fast.
- Audio decoding uses FFmpegKit (`com.antonkarpenko:ffmpeg-kit-audio`) — the
  same FFmpeg command as
  [flutter-examples/vad-from-file](../../flutter-examples/vad-from-file).
- The whole-file preview uses `androidx.media3` ExoPlayer; per-word playback
  uses `AudioTrack` on the decoded PCM (sample-accurate, matches the
  alignment timestamps exactly).

## Build it yourself

See [Build it yourself](../../scripts/mfa/android.md#how-to-build-it-yourself)
in `scripts/mfa/android.md`. In short: build `libsherpa-onnx-jni.so` with
`./build-android-<abi>.sh`, extract a model package into
`app/src/main/assets/`, set `SELECTED_MODEL_INDEX` in `ModelConfig.kt`, copy
the `.so` into `app/src/main/jniLibs/<abi>/`, then:

```bash
./gradlew build -Pabi=arm64-v8a
```

The CI builds all model/ABI combinations with
[scripts/apk/build-apk-forced-alignment.sh.in](../../scripts/apk/build-apk-forced-alignment.sh.in)
via [.github/workflows/apk-fa.yaml](../../.github/workflows/apk-fa.yaml).
