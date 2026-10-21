# Uzbek GigaAM CI fixture

`recording.wav` is a short Uzbek speech recording contributed from
muhammadjonsaidov's `voice-test` project (`recordings/CLIP-002.wav`) for testing
the Python GigaAM example. The contributor authorized its use as CI test data.
It is a mono, 16 kHz, 16-bit PCM WAV file.

`transcript.txt` contains the transcript previously saved by that project's
Java recognizer and reproduced by the Python example. CI compares words after
ignoring case and punctuation. This is a single-clip regression test, not an
accuracy benchmark or an independently annotated reference corpus.

Only the recording and transcript are included; model weights are downloaded
separately from the pinned third-party export by the CI script.
