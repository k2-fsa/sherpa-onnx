# Qwen3-ASR cumulative-audio simulated streaming

This core C++ API reuses the offline Qwen3-ASR graphs. It supports CPU greedy
decoding, tested with 0.6B all-logits FP32 and INT8 models. It is separate from
the C API and `cxx-api.h` wrapper. No model re-export is required.

Each 2-second boundary reruns all accumulated audio with a fresh decoder KV
cache. The first two successful rounds use no prefix. Later rounds retain the
previous raw assistant token IDs minus five tokens, retreating further when
necessary to reach a strict UTF-8 boundary. Display cleanup and homophone
replacement never feed back into that state. Every partial is a complete
hypothesis and can change throughout; retained text is not guaranteed stable.

The caller supplies 16 kHz mono float PCM in [-1,1] and an explicit utterance
end. Resample continuously before packetization if needed. Streams hold the
model alive after the recognizer is destroyed. Serialize all calls that share
an engine, including calls on different streams. Reset retains stream options.

```cpp
#include <algorithm>
#include <stdexcept>
#include <vector>
#include "sherpa-onnx/csrc/qwen3-asr-simulated-streaming.h"

void Transcribe(const sherpa_onnx::OfflineRecognizerConfig &model,
                const std::vector<float> &pcm) {
  using namespace sherpa_onnx;
  Qwen3ASRSimulatedStreamingRecognizer recognizer(model);
  Qwen3ASRSimulatedStreamingOptions options;
  options.language = "Chinese";  // Empty: automatic; use canonical names.
  // options.hotwords = "";      // Disable inherited model hotwords.
  auto stream = recognizer.CreateStream(options);
  auto consume = [](const Qwen3ASRStreamingOutcome &out) {
    for (const auto &update : out.updates) {
      // Replace displayed text with update.result.text, even when empty.
      // update.audio_end_sample/revision identify its audio coverage/version.
      // update.is_final marks the one final result.
    }
    if (out.status != Qwen3ASRStreamingStatus::kOk) {
      throw std::runtime_error("Recognition failed");
    }
  };
  for (size_t i = 0; i < pcm.size();) {
    int32_t n = static_cast<int32_t>(std::min<size_t>(320, pcm.size() - i));
    consume(stream->AcceptWaveform(pcm.data() + i, n));
    i += n;
  }
  consume(stream->Finish());
  auto final_result = stream->GetResult();  // Value copy, also valid after Finish.
  stream->Reset();
}
```

An incomplete final chunk is decoded once. Exact-boundary Finish reuses the
last successful partial and emits one final update. Empty Finish emits an empty
final without inference. Repeated Finish emits no updates. Accept after Finish
returns `kInputFinished`. Reset starts a new utterance.

Invalid samples or a packet exceeding the configured audio limit are rejected
atomically; the stream remains usable. `accepted_samples` and
`processed_samples` distinguish received audio from successfully decoded audio.
If a later snapshot fails within one Accept call, earlier successful updates
are returned, and the last successful hypothesis remains queryable. Further
Accept/Finish calls return the same failure until Reset.

Nonempty prefixes may end immediately with EOS without adding text. A complete
prompt, including audio, language, hotwords and retained IDs, must leave at
least one generation position. Capacity, generation-limit and repetition
stops are explicit failures; they never silently clip new audio or become
successful final results. Short inputs with insufficient features, digital
silence (uniform frontend log floor), and modeled silence clear the hypothesis.
These checks are not a VAD and do not guarantee suppression of noise or
prompted hallucinations.

Strict continuation recognizes Qwen3-ASR's `<|im_end|>` and `<|endoftext|>`
end tokens, resolving their IDs from the tokenizer. Neither end token is
appended to the hypothesis or reused as prefix state. The existing first-token
fallback for an empty prefix is retained; if its replacement is another EOS,
decoding ends without adding it. Ordinary offline decoding retains its original
single-EOS policy and output behavior.

The default and maximum utterance length is 480000 samples (30 s). This is a
resource bound, not a guarantee that every prompt fits model context. New
configuration/options throw `std::invalid_argument`; hotwords are bounded to
16384 UTF-8 bytes and cannot contain control bytes or prompt control tokens.
Model files are trusted inputs. Existing fatal initialization errors in the
core library are not converted; catchable decode exceptions become
`kDecodeError`. Invalid model-output UTF-8 is displayed with one U+FFFD per
invalid byte and flagged, rather than dropped.

## Build and replay

```sh
cmake -S . -B build-prefix -DCMAKE_BUILD_TYPE=Release \
  -DSHERPA_ONNX_ENABLE_TESTS=ON -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
  -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
  -DSHERPA_ONNX_ENABLE_TTS=OFF -DSHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION=OFF
cmake --build build-prefix --target sherpa-onnx-qwen3-asr-simulated-streaming -j4
build-prefix/bin/sherpa-onnx-qwen3-asr-simulated-streaming \
  --qwen3-asr-conv-frontend=/model/conv_frontend.onnx \
  --qwen3-asr-encoder=/model/encoder.int8.onnx \
  --qwen3-asr-decoder=/model/decoder.int8.onnx \
  --qwen3-asr-tokenizer=/model/tokenizer --num-threads=4 \
  --prefix-mode=rollback --packet-size-samples=320 audio.wav
```

The CLI accepts canonical PCM16 mono 16 kHz WAV files and rejects other formats.
It prints JSONL with all updates and explicit failures, exiting nonzero on any
file failure. Multiple files are processed serially with separate streams.
`--prefix-mode=none` sets `unfixed_chunk_num=INT32_MAX` using the same scheduler
and strict decode rules. Effective configuration goes to stderr. `--trace=true`
adds measured service times; default replay is as fast as possible. CLI output
timestamps are not real audio arrival times.

Outcomes also carry `steps`, one diagnostic entry per attempted decode,
including decode exceptions: prompt/retained/generated counts, context limit, stop
reason, full snapshot service time, encoder/decoder time and decoder call
count. Exact-boundary finalization has no new step. Encoder timing includes
feature retrieval, normalization and the convolution frontend; decoder timing
includes prompt construction and sampling. Full snapshot service additionally
includes rollback and waveform feature extraction. No raw IDs are included,
and these measurements never drive recognition state.

Portable control-flow tests require NumPy and ONNX only for fixture generation:

```sh
cmake --build build-prefix --target offline-recognizer-qwen3-asr-impl-test \
  qwen3-asr-simulated-streaming-test -j4
python scripts/qwen3-asr/test-continuation.py --output /tmp/new-qwen-fixtures \
  --binary build-prefix/bin/offline-recognizer-qwen3-asr-impl-test
python scripts/qwen3-asr/test-streaming-cli.py --fixtures /tmp/new-qwen-fixtures \
  --binary build-prefix/bin/sherpa-onnx-qwen3-asr-simulated-streaming
ctest --test-dir build-prefix --output-on-failure
```

The independent reference schedule is the [official Qwen implementation](https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/qwen_asr/inference/qwen3_asr.py).
This API retains raw IDs instead of re-encoding text, applies one UTF-8-safe
rollback rule to regular and final chunks, and reports capacity failures.
It makes no claim of identical vLLM model outputs. Cumulative recomputation can
increase total CPU work; it does not provide constant per-chunk cost or unlimited
duration. Sliding windows, cross-chunk KV reuse, timestamps, VAD and language
bindings are outside this implementation.
