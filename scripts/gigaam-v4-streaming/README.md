# GigaAM-v4 streaming RNN-T

This CPU recognizer uses the [GigaAM-v4 640 ms ONNX package](https://huggingface.co/ai-babai/gigaam-v4-onnx/tree/d39e2a200f596315acef7003d246b65a73e9a327/rnnt-streaming-640ms).
It keeps acoustic and predictor state in each stream. Keep the published
`*_stream_frontend.onnx` file beside the `*_stream_step.int8.onnx` encoder.

Download the pinned release files and export the E2E SentencePiece vocabulary
to Sherpa's token-table format. SentencePiece is used only for this preparation
step. Set `head=raw` to prepare the other vocabulary.

```sh
hf download ai-babai/gigaam-v4-onnx \
  --revision d39e2a200f596315acef7003d246b65a73e9a327 \
  --include 'rnnt-streaming-640ms/*' --local-dir gigaam-v4
cd gigaam-v4/rnnt-streaming-640ms

python -m pip install 'sentencepiece==0.2.2'
head=e2e
python -c '
import sys
from sentencepiece import SentencePieceProcessor
sp = SentencePieceProcessor(model_file=sys.argv[1])
for i in range(sp.get_piece_size()):
    print(sp.id_to_piece(i), i)
print("<blk>", sp.get_piece_size())
' "v4_rnnt_640ms_${head}_tokenizer.model" > "tokens.${head}.txt"
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

E2E returns formatted text. For raw text, use the raw token table with the
`*_raw_decoder.onnx` and `*_raw_joint.onnx` files. The frontend and encoder
are shared. Each recognizer runs one head.

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
