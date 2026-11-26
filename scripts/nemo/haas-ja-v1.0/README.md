---
license: apache-2.0
language:
  - ja
pipeline_tag: automatic-speech-recognition
library_name: sherpa-onnx
base_model: third-intelligence/Haas-ja-v1.0
tags:
  - tdt
  - onnx
  - nemo
  - sherpa-onnx
---

# sherpa-onnx-nemo-haas-ja-v1.0

This package provides [Third Intelligence's Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0) for sherpa-onnx. Third Intelligence trained the weights; the sherpa-onnx project exported them from the original `.nemo` checkpoint. It is one offline transducer (encoder, decoder and joiner) for 16 kHz mono Japanese audio, published as `sherpa-onnx-nemo-haas-ja-v1.0` (fp32) and `sherpa-onnx-nemo-haas-ja-v1.0-int8` (int8).

Haas-ja-v1.0 is a Parakeet-TDT model whose prediction network uses Kimi Delta Attention instead of an LSTM. The exported decoder runs that prediction network one label at a time; its two states have shape `(2, N, 20480)` instead of `(2, N, 640)`, and the encoder metadata says so in `decoder_state_dim`.

## Files

The fp32 package `sherpa-onnx-nemo-haas-ja-v1.0/` contains:

- `encoder.onnx`, with its external weights in `encoder.weights`
- `decoder.onnx`
- `joiner.onnx`
- `tokens.txt`
- `bpe.vocab`
- `test_wavs/`

The int8 package `sherpa-onnx-nemo-haas-ja-v1.0-int8/` contains the quantised graphs instead:

- `encoder.int8.onnx`
- `decoder.int8.onnx`
- `joiner.int8.onnx`
- `tokens.txt`
- `bpe.vocab`
- `test_wavs/`

## Usage

Input must be 16 kHz mono audio. Set `d` to the package directory you downloaded:

```python
import sherpa_onnx
import soundfile as sf

d = "sherpa-onnx-nemo-haas-ja-v1.0-int8"  # or the fp32 package; drop .int8 below
recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder=f"{d}/encoder.int8.onnx",
    decoder=f"{d}/decoder.int8.onnx",
    joiner=f"{d}/joiner.int8.onnx",
    tokens=f"{d}/tokens.txt",
    model_type="nemo_transducer",
)

audio, sample_rate = sf.read("audio.wav", dtype="float32", always_2d=True)
assert sample_rate == 16000, sample_rate
stream = recognizer.create_stream()
stream.accept_waveform(16000, audio[:, 0])
recognizer.decode_stream(stream)
print(stream.result.text)
```

sherpa-onnx reads `decoder_state_dim` and `is_tdt` from the encoder metadata, so versions that do not know these keys cannot load this model.

## Evaluation

FLEURS Japanese test split, all 650 utterances. Character error rate (CER) in %, after NFKC normalization and with punctuation, symbols and spaces removed. Greedy search. These are this project's own numbers, not the model author's, so they are not directly comparable to published benchmarks.

| Model | CER |
| --- | --- |
| Original model in NeMo, bfloat16 on a GPU | 4.59 |
| Original model in NeMo, float32 on a GPU | 4.57 |
| This package, fp32, `sherpa-onnx-offline` | 4.67 |
| This package, int8, `sherpa-onnx-offline` | 4.78 |

Given the same encoder output, the exported decoder and joiner give the same tokens as NeMo for all 650 utterances. The remaining difference comes from the features, which sherpa-onnx computes with kaldi-native-fbank.

## Limitations

- The model is offline (full context). Split long audio at silences; the authors' `transcribe_longform` splits at quiet points into pieces of at most 30 s.
- The int8 package is smaller and faster. The fp32 package is the reference, and its output can differ slightly.
- The original model's custom vocabulary (phrase boosting) is not part of this export. sherpa-onnx hotwords work with `bpe.vocab` and modified beam search.

## Attribution

The weights are Third Intelligence's Haas-ja-v1.0, licensed under [Apache-2.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0). This ONNX export was produced by the sherpa-onnx project, whose code is licensed under [Apache-2.0](https://github.com/k2-fsa/sherpa-onnx/blob/master/LICENSE). Third Intelligence does not endorse this package.

## Export

The export script is [`scripts/nemo/haas-ja-v1.0/`](https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/nemo/haas-ja-v1.0). The package is built and published by [`.github/workflows/export-nemo-haas-ja-v1.0.yaml`](https://github.com/k2-fsa/sherpa-onnx/blob/master/.github/workflows/export-nemo-haas-ja-v1.0.yaml).
