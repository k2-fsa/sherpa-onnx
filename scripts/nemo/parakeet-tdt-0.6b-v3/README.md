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
base_model: nvidia/parakeet-tdt-0.6b-v3
tags:
  - parakeet
  - tdt
  - onnx
  - nemo
  - sherpa-onnx
---

# sherpa-onnx-nemo-parakeet-tdt-0.6b-v3

This package provides [NVIDIA's parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) for sherpa-onnx. NVIDIA trained the weights; the sherpa-onnx project exported them from the original `.nemo` checkpoint. It is one offline transducer — encoder, decoder and joiner — for 16 kHz mono audio in 25 languages, published as `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3` (fp32) and `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8` (int8).

## Files

The fp32 package `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3/` contains:

- `encoder.onnx`, with its external weights in `encoder.weights`
- `decoder.onnx`
- `joiner.onnx`
- `tokens.txt`
- `test_wavs/`

The int8 package `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/` contains the quantised graphs instead:

- `encoder.int8.onnx`
- `decoder.int8.onnx`
- `joiner.int8.onnx`
- `tokens.txt`
- `test_wavs/`

## Usage

Input must be 16 kHz mono audio. Set `d` to the package directory you downloaded:

```python
import sherpa_onnx
import soundfile as sf

d = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"  # or the fp32 package; drop .int8 below
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

| Language | v3 (this package) |
| --- | --- |
| English | 11.16 |
| German | 6.84 |
| French | 8.69 |
| Spanish | 4.55 |

For comparison, v2 covers English only and measures 6.48 there, so v2 remains the strongest of the three in English. Ultra, the other multilingual package, measures 5.20 German, 7.48 French and 3.81 Spanish against v3's numbers above, so Ultra is the strongest of the three in German, French and Spanish.

Short English clips of 15 s or less (282 clips): v2 6.42, v3 11.01, Ultra 8.56.

## Limitations

- **Long audio must be split at silences.** Decoding a whole recording as one clip degrades badly and grows memory. In a stress test with concatenated FLEURS English sentences and 0.2 s silences between them, whole-clip decoding reached 66.50% WER on a 184.6 s clip with +1645 MB RSS over the loaded-model baseline; splitting at silences into 15 s chunks reached 14.91% WER with +142 MB. This is a stress test, not natural long-form speech, and these are comparative measurements on one Linux CPU machine, not universal. Split at silences into 15-20 s chunks; never use fixed windows.
- The int8 package is smaller and faster and is what most users want. The fp32 package is the reference, and its output can differ slightly.
- The model transcribes; it does not report which language it heard, and it is not a language identifier.
- Very quiet input is affected by a frontend issue tracked upstream ([k2-fsa/sherpa-onnx#4024](https://github.com/k2-fsa/sherpa-onnx/issues/4024)); normalise gain before the model sees it.
- English is where the three models differ most. Pick v2 for English if accuracy on longer utterances matters more than one package for every language.

## Attribution

The weights are NVIDIA's parakeet-tdt-0.6b-v3, trained by NVIDIA and licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This ONNX export was produced by the sherpa-onnx project, whose code is licensed under [Apache-2.0](https://github.com/k2-fsa/sherpa-onnx/blob/master/LICENSE). NVIDIA does not endorse this package.

## Export

The export script is [`scripts/nemo/parakeet-tdt-0.6b-v3/`](https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/nemo/parakeet-tdt-0.6b-v3). The package is built and published by [`.github/workflows/export-nemo-parakeet-tdt-0.6b-v2.yaml`](https://github.com/k2-fsa/sherpa-onnx/blob/master/.github/workflows/export-nemo-parakeet-tdt-0.6b-v2.yaml).
