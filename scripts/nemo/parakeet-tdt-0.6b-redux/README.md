---
license: cc-by-4.0
language:
  - bg
  - hr
  - cs
  - da
  - nl
  - en
  - et
  - fi
  - fr
  - de
  - el
  - hu
  - it
  - lv
  - lt
  - mt
  - pl
  - pt
  - ro
  - sk
  - sl
  - es
  - sv
  - ru
  - uk
pipeline_tag: automatic-speech-recognition
library_name: sherpa-onnx
base_model: [moondream/parakeet-redux, nvidia/parakeet-tdt-0.6b-v3]
tags:
  - parakeet
  - tdt
  - onnx
  - nemo
  - sherpa-onnx
---

# sherpa-onnx-nemo-parakeet-tdt-0.6b-redux

This package provides [Moondream's Parakeet Redux](https://huggingface.co/moondream/parakeet-redux) for sherpa-onnx. Redux is a 1.58-bit version of [NVIDIA's parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3): the same architecture, the same 8193-token vocabulary, but every encoder linear and pointwise-conv weight is -1, 0 or +1 with a scale per group of 128 columns. Moondream ships those weights packed, and the packing is what makes the checkpoint 178 MB. ONNX Runtime has no ternary kernel, so this export decodes the packing into a dense graph and then quantises it: the package here is a normal ONNX transducer, not a 178 MB one.

It is one offline transducer — encoder, decoder and joiner — for 16 kHz mono audio in 25 languages, published as `sherpa-onnx-nemo-parakeet-tdt-0.6b-redux` (fp32) and `sherpa-onnx-nemo-parakeet-tdt-0.6b-redux-int8` (int8).

## Files

The fp32 package `sherpa-onnx-nemo-parakeet-tdt-0.6b-redux/` contains:

- `encoder.onnx`, with its external weights in `encoder.weights`
- `decoder.onnx`
- `joiner.onnx`
- `tokens.txt`
- `test_wavs/`

The int8 package `sherpa-onnx-nemo-parakeet-tdt-0.6b-redux-int8/` contains the quantised graphs instead:

- `encoder.int8.onnx`
- `decoder.int8.onnx`
- `joiner.int8.onnx`
- `tokens.txt`
- `test_wavs/`

The two packages hold the same weights at different precision. The int8 tarball is 327,736,417 bytes against 487,131,623 for the Ultra int8 package, because the ternary alphabet survives int8 quantisation as very few distinct values and bzip2 compresses it well. Unpacked, both are about 670 MB; the saving is in the download, not on disk.

## Usage

Input must be 16 kHz mono audio. Set `d` to the package directory you downloaded:

```python
import sherpa_onnx
import soundfile as sf

d = "sherpa-onnx-nemo-parakeet-tdt-0.6b-redux-int8"  # or the fp32 package; drop .int8 below
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

The encoder metadata written by the exporter contains `tdt` in its `url` field, so sherpa-onnx loads the joiner as a TDT joiner rather than a plain transducer joiner.

## Evaluation

FLEURS test split, 300 deterministic clips per language, punctuation and case stripped, decoded with one harness for all three models; word error rate (WER) in %. These are this project's own numbers, not the model author's, so they are not directly comparable to published benchmarks.

| Language | Redux int8 (this package) | Redux fp32 | v3 int8 | Ultra int8 |
| --- | --- | --- | --- | --- |
| English | 13.86 | 7.62 | 11.16 | 8.51 |
| German | 11.00 | — | 6.84 | 5.20 |
| French | 10.97 | — | 8.69 | 7.48 |
| Spanish | 5.04 | — | 4.55 | 3.81 |
| Maltese | 39.74 | — | 56.41 | 44.74 |
| Greek | 49.46 | — | 46.64 | 45.19 |
| **average of the six** | **21.68** | — | **22.38** | **19.16** |

Redux is ahead of v3 on Maltese by 16.7 points and behind it on English, German, French and Greek; across these six languages its average is slightly better. It is behind Ultra in every language above except Maltese. Pick it when the smaller download matters or the language is one where it wins, not as a general replacement for Ultra.

## Limitations

- **The int8 package costs more accuracy here than it does for v3 or Ultra.** On English, Redux measures 7.62% at fp32 and 13.86% at int8, while Ultra measures 5.33% and 8.51%. The ternary weights carry a scale per group of 128 columns, which a per-channel dynamic quantiser cannot represent; the quantisation error on the weights themselves is small, but the encoder output diverges and the transcript loses accuracy. Use the fp32 package when accuracy matters and the disk allows it.
- **Long audio must be split at silences.** Decoding a whole recording as one clip degrades and grows memory; split at silences into 15-20 s chunks, never fixed windows. This is inherited from the v3 offline decoder and is not specific to Redux.
- The model transcribes; it does not report which language it heard, and it is not a language identifier.
- Very quiet input is affected by a frontend issue tracked upstream ([k2-fsa/sherpa-onnx#4024](https://github.com/k2-fsa/sherpa-onnx/issues/4024)); normalise gain before the model sees it.

## Attribution

The weights were trained by Moondream from NVIDIA's parakeet-tdt-0.6b-v3 architecture and tokenizer, and are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This ONNX export was produced by the sherpa-onnx project, whose code is licensed under [Apache-2.0](https://github.com/k2-fsa/sherpa-onnx/blob/master/LICENSE). Neither NVIDIA nor Moondream endorses this package.

## Export

The export script is [`scripts/nemo/parakeet-tdt-0.6b-redux/`](https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/nemo/parakeet-tdt-0.6b-redux). The package is built and published by [`.github/workflows/export-nemo-parakeet-tdt-0.6b-v2.yaml`](https://github.com/k2-fsa/sherpa-onnx/blob/master/.github/workflows/export-nemo-parakeet-tdt-0.6b-v2.yaml). The packing is decoded with the rule documented by the checkpoint's own `ternary.json` (`thrush-ternary-v2`), and `test_redux_regression.py` in that directory pins the decode, the conv gate and the package wiring. The request for this package is [k2-fsa/sherpa-onnx#4161](https://github.com/k2-fsa/sherpa-onnx/issues/4161).
