# MFA forced-alignment models for sherpa-onnx-fa

This directory packages GMM acoustic models trained with the
[Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io)
(MFA) for use with the forced-alignment support of
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx). Each model directory is
self-contained: it downloads the upstream model, generates a pronunciation
lexicon for it and packs `<name>.tar.bz2` for the
[fa-models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/fa-models)
release.

## Table of contents

- [English](#english)
  - [Sources](#english-sources)
- [Chinese](#chinese)
  - [Sources](#chinese-sources)
- [Model package contents](#model-package-contents)
- [Lexicon format](#lexicon-format)
- [Usage](#usage)
- [How the packages are built](#how-the-packages-are-built)

## English

English MFA acoustic models of
[MontrealCorpusTools/mfa-models](https://github.com/MontrealCorpusTools/mfa-models):

| Model | Download | Phones | MFA dictionary | lexicon.txt | Directory | Source |
|---|---|---|---|---|---|---|
| english_mfa v2.2.1 | [english-mfa-v2.2.1.tar.bz2](https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/english-mfa-v2.2.1.tar.bz2) | 100 | v2.2.1, 95k words | ~204k lines | [`en/v2.2.1/english-mfa`](en/v2.2.1/english-mfa) | [acoustic-english_mfa-v2.2.1](https://github.com/MontrealCorpusTools/mfa-models/releases/tag/acoustic-english_mfa-v2.2.1) |
| english_mfa v3.0.0 | [english-mfa-v3.0.0.tar.bz2](https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/english-mfa-v3.0.0.tar.bz2) | 100 | v3.0.0, 95k words | ~204k lines | [`en/v3.0.0/english-mfa`](en/v3.0.0/english-mfa) | [acoustic-english_mfa-v3.0.0](https://github.com/MontrealCorpusTools/mfa-models/releases/tag/acoustic-english_mfa-v3.0.0) |
| english_mfa v3.1.0 | [english-mfa-v3.1.0.tar.bz2](https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/english-mfa-v3.1.0.tar.bz2) | 100 | v3.1.0, 42k words | ~43k lines | [`en/v3.1.0/english-mfa`](en/v3.1.0/english-mfa) | [acoustic-english_mfa-v3.1.0](https://github.com/MontrealCorpusTools/mfa-models/releases/tag/acoustic-english_mfa-v3.1.0) |

Each version is packaged by its [`en/<version>/english-mfa`](en) directory
(`run.sh` + `generate_lexicon.py`, details in `notes.md`); see also
[`en/README.md`](en/README.md) for how the English lexicons are generated.
What differs between the versions:

- **Phone set**: v2.2.1 and later use the 100-phone MFA IPA set (v2.0.0
  used an older 79-phone set and is not packaged here: those models ship no
  `phones.txt` and their `meta.json` phone list does not match the model's
  phone set, so their phone ids cannot be recovered and `sherpa-onnx-fa`
  rejects them). A lexicon must match the phone set of its model, so each
  version ships its own `lexicon.txt`.
- **Dictionary**: every version is paired with the dictionary of the same
  version. v3.1.0's dictionary is a smaller core vocabulary (42k words);
  the others cover ~95k words. `lexicon.txt` is essentially these
  pronunciations (one line per pronunciation, so multi-pronunciation words
  take several lines) plus the phrase entries, which is why its size
  follows the dictionary size (~43k lines for v3.1.0 vs ~200k for the
  others). `lexicon.txt` additionally lists the words of the English word
  list (see below), and multi-word phrases with their combined readings.
- **Word list**: `lexicon.txt` covers the words of
  [lexicon-en-us.txt](https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/lexicon-en-us.txt)
  (words + the words of its phrases) unioned with the dictionary's own
  words. Pronunciations come from the MFA dictionary; multi-word phrases
  keep their combined readings (they differ from the concatenation of the
  word readings, like polyphones). Words without a pronunciation are left
  out and the aligner reports them as out-of-vocabulary at runtime.

### English sources

- **Acoustic models**: the `english_mfa.zip` archives of the
  [acoustic-english_mfa-vX.Y.Z](https://github.com/MontrealCorpusTools/mfa-models/releases)
  releases (upstream name: `english_mfa`; GMM-HMM models trained with the
  Montreal Forced Aligner, see the upstream model cards for training
  details).
- **Dictionaries**: the `english_mfa.dict` archives of the
  [dictionary-english_mfa-vX.Y.Z](https://github.com/MontrealCorpusTools/mfa-models/releases)
  releases of the same repository, each paired with the model of the same
  version.
- **Word list**:
  [lexicon-en-us.txt](https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/lexicon-en-us.txt).

## Chinese

Chinese MFA acoustic models of
[lars76/forced-alignment-chinese](https://github.com/lars76/forced-alignment-chinese):

| Model | Download | Training data | Directory | Source |
|---|---|---|---|---|
| aishell3_pinyin | [zh-aishell3-pinyin.tar.bz2](https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/zh-aishell3-pinyin.tar.bz2) | AISHELL-3 | [`zh/lars76/aishell3_pinyin`](zh/lars76/aishell3_pinyin) | [model_files](https://github.com/lars76/forced-alignment-chinese/releases/tag/model_files) |
| biaobei_pinyin | [zh-biaobei-pinyin.tar.bz2](https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/zh-biaobei-pinyin.tar.bz2) | biaobei reading corpus | [`zh/lars76/biaobei_pinyin`](zh/lars76/biaobei_pinyin) | [model_files](https://github.com/lars76/forced-alignment-chinese/releases/tag/model_files) |
| large_pinyin | [zh-large-pinyin.tar.bz2](https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/zh-large-pinyin.tar.bz2) | AISHELL-3 + biaobei + more | [`zh/lars76/large_pinyin`](zh/lars76/large_pinyin) | [model_files](https://github.com/lars76/forced-alignment-chinese/releases/tag/model_files) |

Each model is packaged by its [`zh/lars76/<name>`](zh/lars76) directory
(`run.sh` + a shared `generate_lexicon.py`, details in `notes.md`); the
`zh-<name>.tar.bz2` archives are assembled by
[`zh/lars76/package-fa-model.sh`](zh/lars76/package-fa-model.sh). The
language prefix in the package names keeps them unambiguous now that
models for other languages are included.

### Chinese sources

- **Acoustic models**: the `<name>.zip` bundles of the
  [model_files](https://github.com/lars76/forced-alignment-chinese/releases/tag/model_files)
  release, each containing the acoustic model (a nested `*_acoustic.zip`
  with `meta.json`/`phones.txt`/`final.alimdl`/`tree`/`lda.mat`) and its
  pronunciation dictionary.
- **Pronunciations** for `lexicon.txt` are generated from those
  dictionaries (pinyin-to-phone conversion via `pypinyin` +
  `pinyin_to_ipa`); no pronunciation data comes from anywhere else.

## Model package contents

Each package is a single model directory holding only the files required by
`sherpa-onnx-fa`:

```
<name>/meta.json      feature chain + silence/OOV parameters
<name>/final.alimdl   alignment model (kaldi TransitionModel + AmDiagGmm)
<name>/tree           context dependency tree
<name>/lda.mat        LDA transform of the feature chain
<name>/phones.txt     phone table (when the upstream model ships one)
<name>/lexicon.txt    pronunciation dictionary (generated here)
<name>/en.wav         test wav (en.wav for English models, zh.wav for
                      Chinese ones; its transcript is in README.md)
<name>/README.md      model provenance (where the model comes from) and
                      a usage example that runs as-is on the test wav
```

Extra files of the upstream bundles (`final.mdl`, `phone_lm.fst`,
`graphemes.txt`, `phone_pdf.counts`, the pronunciation dictionaries, ...)
are dropped. All models follow whatever license their upstream declares; the
English model cards state CC BY 4.0 (see the source repositories).

## Lexicon format

`lexicon.txt` is a pronunciation dictionary, one pronunciation per line:

```
word [probability columns ...] phone1 phone2 ...
```

Probability columns are optional and ignored. Several lines under one key =
several pronunciations; the aligner keeps them all as parallel paths and the
acoustics pick the best one. Words without a pronunciation are left out of
the lexicon and reported as out-of-vocabulary at runtime.

- **English**: keys are words; multi-word phrases keep their combined
  readings, ` || ` separating the per-word phone groups.
- **Chinese**: keys are hanzi (characters and multi-character phrases),
  phones are pinyin-derived; ` || ` separates the per-character phone
  groups. Multi-character keys provide pronunciation context for
  polyphones. No runtime pypinyin is needed.

## Usage

### Pre-built binary

`sherpa-onnx-fa`, the command line tool shown below, is only a demo. The
models are not tied to it: forced alignment is part of the sherpa-onnx
libraries and the same models work through the APIs of all supported
programming languages (C, C++, Python, Java/Kotlin, C#, Go, Swift, Rust,
JavaScript/Node.js, Dart/Flutter, HarmonyOS, ...), see the
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) documentation of the
language of your choice.

```bash
sherpa-onnx-fa \
  --model-dir=/path/to/<name> \
  --lexicon=/path/to/<name>/lexicon.txt \
  --wav=input.wav \
  --text="..." \
  --out=output.TextGrid
```

The output is a Praat TextGrid with a phone tier and a word tier. Open it
with [Praat](https://www.fon.hum.uva.nl/praat/) to visualize the phone and
word alignments of the utterance. For hanzi text the words tier is one word
per character (multi-character keys provide pronunciation context);
otherwise whitespace tokenization is used.

Each package contains a test wav (`en.wav` for English models, `zh.wav` for
Chinese ones) whose transcript is written out in the model's own `README.md`
— its usage example runs as-is on the extracted package.

### Python API

The same models work through the sherpa-onnx Python API (see `help()` on
the classes for the full documentation):

```python
import sherpa_onnx

config = sherpa_onnx.ForcedAlignerConfig()
config.model_dir = "english-mfa-v3.1.0"
config.lexicon = "english-mfa-v3.1.0/lexicon.txt"

aligner = sherpa_onnx.ForcedAligner(config)
s = aligner.create_stream()
s.accept_waveform(sample_rate, samples)  # float32, normalized to [-1, 1]
result = aligner.compute(s, text)
print(result.to_textgrid())
```

Complete runnable examples (with the download commands in their headers)
are in
[python-api-examples](../../python-api-examples): `forced-alignment-english.py`,
`forced-alignment-chinese.py` and `textgrid-to-word-wavs.py` (cutting the
segment of each word).

## How the packages are built

Every model directory has a one-click `run.sh`:

```bash
cd en/v3.1.0/english-mfa && ./run.sh     # English: downloads + lexicon + package
cd zh/lars76/aishell3_pinyin && ./run.sh # Chinese: downloads + lexicon
```

The Chinese packages are assembled by
[`zh/lars76/package-fa-model.sh`](zh/lars76/package-fa-model.sh)
(`package-fa-model.sh <name>`), which also writes the provenance `README.md`
of the model directory.

The
[upload-fa-models](../../.github/workflows/upload-fa-models.yaml)
workflow runs the packaging on CI and uploads all six `*.tar.bz2` archives
to the [fa-models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/fa-models)
release. The archives in the directories above are build outputs of those
scripts (regenerable, not sources).
