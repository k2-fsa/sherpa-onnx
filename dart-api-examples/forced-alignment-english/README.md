# Forced alignment (English)

This example shows how to use the Dart API to align a transcript to an
English utterance and get the phone and word boundaries.

It also writes `en.TextGrid`, a
[Praat](https://www.fon.hum.uva.nl/praat/) TextGrid you can open to
visualize the boundaries.

## Usage

```bash
./run.sh
```

`run.sh` downloads the model from the
[fa-models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/fa-models)
release if it is missing.

The model directory must contain `meta.json`, `final.alimdl`, `tree` (and
`lda.mat` iff the feature chain is splice+LDA). The lexicon file has one
pronunciation per line: `key [probability columns ...] phone1 phone2 ...`.
