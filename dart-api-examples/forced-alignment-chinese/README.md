# Forced alignment (Chinese)

This example shows how to use the Dart API to align a transcript to a
Chinese utterance and get the phone and word boundaries. The words tier has
one interval per character.

It also writes `zh.TextGrid`, a
[Praat](https://www.fon.hum.uva.nl/praat/) TextGrid you can open to
visualize the boundaries.

## Usage

```bash
./run.sh
```

`run.sh` downloads the model from the
[fa-models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/fa-models)
release if it is missing.

The lexicon is hanzi-keyed; a polyphone keeps all of its pronunciations and
the acoustics pick the one used by the alignment. A multi-token key writes
one ` || `-separated phone group per character, e.g.
`银行 || i n || h a ŋ`.
