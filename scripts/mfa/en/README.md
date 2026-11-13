# English MFA forced-alignment models

This directory packages the English MFA acoustic models of
[MontrealCorpusTools/mfa-models](https://github.com/MontrealCorpusTools/mfa-models)
for [sherpa-onnx-fa](https://github.com/k2-fsa/sherpa-onnx). Each model
directory is self-contained: its `run.sh` downloads the model and its
dictionary, generates `lexicon.txt` and packs `<name>.tar.bz2` for the
[fa-models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/fa-models)
release.

## Versions

| Directory | Model | Phones | MFA dictionary | lexicon.txt |
|---|---|---|---|---|
| `v2.2.1/english-mfa` | [english_mfa v2.2.1](https://github.com/MontrealCorpusTools/mfa-models/releases/tag/acoustic-english_mfa-v2.2.1) | 100 | [v2.2.1](https://github.com/MontrealCorpusTools/mfa-models/releases/tag/dictionary-english_mfa-v2.2.1), 95k words | ~204k lines |
| `v3.0.0/english-mfa` | [english_mfa v3.0.0](https://github.com/MontrealCorpusTools/mfa-models/releases/tag/acoustic-english_mfa-v3.0.0) | 100 | [v3.0.0](https://github.com/MontrealCorpusTools/mfa-models/releases/tag/dictionary-english_mfa-v3.0.0), 95k words | ~204k lines |
| `v3.1.0/english-mfa` | [english_mfa v3.1.0](https://github.com/MontrealCorpusTools/mfa-models/releases/tag/acoustic-english_mfa-v3.1.0) | 100 | [v3.1.0](https://github.com/MontrealCorpusTools/mfa-models/releases/tag/dictionary-english_mfa-v3.1.0), 42k words | ~43k lines |

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
  others). `lexicon.txt` additionally lists the words
  of the English word list (see below), and multi-word phrases with their
  combined readings.
- **Word list**: `lexicon.txt` covers the words of
  [lexicon-en-us.txt](https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/lexicon-en-us.txt)
  (words + the words of its phrases) unioned with the dictionary's own
  words. Pronunciations come from the MFA dictionary; multi-word phrases
  keep their combined readings (they differ from the concatenation of the
  word readings, like polyphones). Words without a pronunciation are left
  out and the aligner reports them as out-of-vocabulary at runtime.

## Usage

```bash
cd v3.1.0/english-mfa && ./run.sh
```

Then:

```
sherpa-onnx-fa \
  --model-dir=english-mfa-v3.1.0 \
  --lexicon=english-mfa-v3.1.0/lexicon.txt \
  --wav=input.wav \
  --text="..." \
  --out=output.TextGrid
```

## License

These models follow whatever license the original models use; the upstream
model cards declare **CC BY 4.0**. See the
[source repository](https://github.com/MontrealCorpusTools/mfa-models) for
the license terms.
