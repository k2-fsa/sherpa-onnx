# Online Silero VAD for the streaming websocket server: design notes

## What it does

An optional Silero VAD runs next to the streaming ASR in `sherpa-onnx-online-websocket-server`
(enabled by `--online-vad-model`). When the VAD confirms that speech has ended, it arms the
endpoint itself and the server decodes the last partial chunk right away, instead of waiting
for the blank-based endpoint rules. With the VAD off, nothing changes.

## Structure

```
OnlineVadModel            abstract: RunVad, WarmUp, SampleRate, WindowShift, WindowSize
  OnlineSileroVad         one shared read-only ONNX session, batched inference
OnlineStream              per-stream VAD state (LSTM state, pending audio, counters,
                          endpoint/flush flags)
OnlineRecognizer          owns the single vad_model_, exposes VadProcess / VadHasWindow /
                          IsDecodable
OnlineWebsocketDecoder    VAD loop + ASR loop
```

- The model is created once per recognizer, not per connection.
- All mutable state lives on the stream, like the ASR encoder state, so one session serves
  every connection.
- New VAD models plug into `OnlineVadModel::Create(config)`.

## Data flow

1. `AcceptWaveform` copies incoming audio into a per-stream `vad_pending_` buffer, resampled
   to 16 kHz. This is separate from the ASR feature path.
2. The VAD loop (`--vad-loop-interval-ms`, `--num-vad-threads`) queues only streams that hold
   a full window (`VadHasWindow`), up to `--max-vad-batch-size` per call. It never waits to
   fill a batch.
3. `OnlineSileroVad::RunVad` stacks the states, runs one batched inference per round and
   applies the threshold / min-speech / min-silence rule to each stream.
4. `OnlineStream::VadPushPrediction` records the result. On a confirmed speech-to-silence
   flip it sets the endpoint and a pending flush up to a given frame.
5. `ScheduleVadFlushes` posts `Decode()` immediately for those streams.
6. `IsReady` treats a pending flush as ready. `DecodeStreams` decodes the real frames up to
   the endpoint, pads the rest of the chunk with silence, and clears the flag.
7. `IsEndpoint` reports the endpoint only after the flush is decoded, so the final words are
   in the result.

Outside a flush, ASR waits until the VAD has covered a chunk, and with
`--online-vad-batch-skip` it skips chunks that lie inside confirmed silence.

## Threading

- ASR and VAD have separate mutexes and loops, so they can work on the same stream at the same
  time. Per-stream VAD state is only touched by the VAD side; the endpoint/flush flags are the
  hand-off to the ASR side.
- `Connection::eof` is `std::atomic<bool>` because the ASR and VAD loops read it under
  different mutexes (found with ThreadSanitizer).

## Quick facts

| Item | Value |
|---|---|
| Model | Silero v5, 16 kHz, window 512 (+64 context) = 32 ms stride |
| Endpoint wait | `--online-vad-min-silence-duration` is most of the remaining latency |
| Batching | the VAD takes what is ready; it never waits to fill a batch |
| Flags | `--online-vad-*`, `--vad-loop-interval-ms`, `--max-vad-batch-size`, `--num-vad-threads` |

## Measured trade-offs

Streaming zipformer transducer, 8 vCPU + A10G, annotated endpoint test, VAD off to on:

- Endpoint latency p50 / p90: c=100 1450/1850 ms to 640/750 ms; c=200 1450/1845 ms to
  660/1110 ms.
- False alarms rise from about 9% to 14%; misses are unchanged.
- The VAD costs CPU (CPU Silero inference). On a CPU-bound server it competes with ASR
  threads, so give it enough `--num-vad-threads` and keep the VAD loop interval short; too few
  VAD threads make ASR wait for VAD coverage.

## Known limitations

- A rare `features.cc:GetFrames` abort was seen once and not reproduced. `IsReady()` could be
  evaluated twice per tick (it can advance processed frames); it is now evaluated once and
  `Decode()` re-checks with the side-effect-free `IsDecodable()`. Not proven to be the cause.
- Endpoint timing assumes the Silero v5, 16 kHz, 512-sample window setup.
- The end-of-utterance flush is implemented for the transducer recognizer only; other model
  types get VAD silence skipping but not the flush.
