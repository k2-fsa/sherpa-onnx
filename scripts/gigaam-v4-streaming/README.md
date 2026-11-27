# GigaAM-v4 streaming RNN-T

This CPU recognizer uses the [GigaAM-v4 640 ms ONNX package](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/00936113c4ccdb679da0212ebdef8e52b73aac48/rnnt-streaming-640ms).
It keeps acoustic and predictor state in each stream. Keep the matching
`*_stream_frontend.onnx` file beside the encoder. The encoder filename can
end in `_stream_step.onnx`, `_stream_step.int8.onnx`,
`_stream_step.int4.onnx`, or `_stream_step.fp16.onnx`. The exported tensor
interface must be the same for each precision.

Use a Sherpa source build with the `gigaam_v4` model type. The examples were
exercised with Sherpa 2.0.3 and ONNX Runtime 1.28.2. Install the full Python
package from your source checkout with `python -m pip install .`.

Download the pinned graphs and both published token tables together.

```sh
python -m pip install 'huggingface_hub==1.3.5'
hf download ai-babai/gigaam-v4-onnx \
  --revision 00936113c4ccdb679da0212ebdef8e52b73aac48 \
  --include 'rnnt-streaming-640ms/*' --local-dir gigaam-v4
cd gigaam-v4/rnnt-streaming-640ms
```

## Python streaming API

Run this example from the model directory with a 16 kHz mono PCM16 WAV named
`speech.wav`. Supply normalized float samples to `accept_waveform`.

```python
import sys
import wave
from array import array

import sherpa_onnx

recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens="tokens.e2e.txt",
    encoder="v4_rnnt_640ms_stream_step.int8.onnx",
    decoder="v4_rnnt_640ms_e2e_decoder.onnx",
    joiner="v4_rnnt_640ms_e2e_joint.onnx",
    model_type="gigaam_v4",
    provider="cpu",
    num_threads=3,
    enable_endpoint_detection=False,
)
stream = recognizer.create_stream()


def drain():
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)


with wave.open("speech.wav", "rb") as wav:
    if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
        raise ValueError("Expected 16 kHz mono PCM16 WAV")
    while block := wav.readframes(10240):
        pcm = array("h", block)
        if sys.byteorder == "big":
            pcm.byteswap()
        stream.accept_waveform(16000, [sample / 32768.0 for sample in pcm])
        drain()
        text = recognizer.get_result(stream)
        if text:
            print("partial", text)

stream.input_finished()
drain()
print("final", recognizer.get_result(stream))
```

`get_result` returns cumulative text for the current utterance.
`recognizer.reset(stream)` clears caches and discards queued samples from the
previous utterance. After EOF, create a new stream to begin another input.
C and C++ clients use the same lifecycle and set `model_config.model_type` to
`gigaam_v4`.

Use `tokens.e2e.txt` with the E2E decoder and joint for formatted text.
Use `tokens.raw.txt` with the raw decoder and joint for plain text. The
frontend and encoder are shared. Each recognizer runs one head.

The recognizer checks the head names on the published decoder, joint, and token
files and rejects a mixed pair. Renamed files without a head suffix and
`tokens_buf` carry no head name. Keep a published head-identifying name on at
least one file for raw output, and supply the matching files or token buffer.
If no name identifies the head, text uses the E2E formatting convention.

## CLI file smoke test

The CLI submits the WAV as one input. Use the Python example to exercise
incremental packet delivery.

```sh
sherpa-onnx --model-type=gigaam_v4 --provider=cpu --num-threads=3 \
  --left-padding=0 --right-padding=0 \
  --decoding-method=greedy_search --enable-endpoint=false \
  --tokens=tokens.e2e.txt \
  --encoder=v4_rnnt_640ms_stream_step.int8.onnx \
  --decoder=v4_rnnt_640ms_e2e_decoder.onnx \
  --joiner=v4_rnnt_640ms_e2e_joint.onnx speech.wav
```

## Streaming regression test

From the source checkout, download the pinned package above without changing
directories. Configure CMake with `SHERPA_ONNX_ENABLE_PYTHON=ON` and
`SHERPA_ONNX_ENABLE_TESTS=ON`, build `_sherpa_onnx`, and install NumPy in
the Python environment used by the build.

```sh
mkdir -p sherpa-onnx/python/sherpa_onnx/lib
cp build/lib/_sherpa_onnx*.so sherpa-onnx/python/sherpa_onnx/lib/
curl -fL -o gigaam-v4/test.wav \
  https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition/resolve/2408de4949741cb03b444a8edbc3d38acd769927/test_wavs/russian/test.wav
export SHERPA_ONNX_GIGAAM_STREAMING_PACKAGE="$PWD/gigaam-v4/rnnt-streaming-640ms"
export SHERPA_ONNX_GIGAAM_TEST_WAV="$PWD/gigaam-v4/test.wav"
ctest --test-dir build -R '^test_online_gigaam_v4_py$' --output-on-failure
```

The WAV SHA-256 is `d6e768803b3bc3afcb08326677f3bb872c9beeed29af40d3579bcc14e74484f8`.
The test checks partial output, EOF draining, a word from the transcript,
and mixed-head rejection.
CTest skips it if the fixture variables are unset. To test a released INT4
package, point the package variable at that directory and set
`SHERPA_ONNX_GIGAAM_STREAMING_ENCODER=v4_rnnt_640ms_stream_step.int4.onnx`.
