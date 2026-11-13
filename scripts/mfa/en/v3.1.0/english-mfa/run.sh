#!/usr/bin/env bash
set -euo pipefail

# Download the english_mfa v3.1.0 acoustic model, its MFA dictionary and
# the English word list, generate lexicon.txt, and package the model
# directory (english-mfa-v3.1.0.tar.bz2) for the fa-models release. See notes.md.

cd "$(dirname "$0")"

pkg=english-mfa-v3.1.0
acoustic_url=https://github.com/MontrealCorpusTools/mfa-models/releases/download/acoustic-english_mfa-v3.1.0/english_mfa.zip
dict_url=https://github.com/MontrealCorpusTools/mfa-models/releases/download/dictionary-english_mfa-v3.1.0/english_mfa.dict
words_url=https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/lexicon-en-us.txt

download_file() {  # url dest
  if [ ! -f "$2" ]; then
    echo "Downloading $1"
    curl -SL --fail --retry 5 --retry-all-errors --connect-timeout 30 \
      -o "$2.tmp" "$1"
    mv "$2.tmp" "$2"
  fi
}

mkdir -p download
if [ ! -f download/.model_extracted ]; then
  download_file "$acoustic_url" download/acoustic.zip
  download_file "$dict_url" download/dictionary.dict
  download_file "$words_url" download/lexicon-en-us.txt
  unzip -q -o download/acoustic.zip -d download
  touch download/.model_extracted
fi

# test wav shipped with the package (transcript in the usage example below)
download_file https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/en.wav download/en.wav

python3 generate_lexicon.py

# ---- package: only the files required by sherpa-onnx-fa, plus lexicon.txt
# and a README.md documenting where the model comes from ----
model_dir=$(dirname "$(find download -name meta.json -type f | head -1)")
stage=package/$pkg
rm -rf package
mkdir -p "$stage"
for f in meta.json final.alimdl tree lda.mat; do
  cp "$model_dir/$f" "$stage/"
done
if [ -f "$model_dir/phones.txt" ]; then
  cp "$model_dir/phones.txt" "$stage/"
fi
cp lexicon.txt "$stage/"
cp download/en.wav "$stage/"

cat > "$stage/README.md" <<EOF
# english-mfa-v3.1.0 - forced-alignment acoustic model

This is the english_mfa acoustic model (version v3.1.0) for forced
alignment, as used by sherpa-onnx-fa of
https://github.com/k2-fsa/sherpa-onnx.

## Where the model comes from

- Model archive: $acoustic_url
  (upstream name: english_mfa)
- Source repository: https://github.com/MontrealCorpusTools/mfa-models
  (GMM-HMM acoustic model trained with the Montreal Forced Aligner - see
  the upstream README for training details)

This directory is a re-packaging made by
scripts/mfa/en/v3.1.0/english-mfa/run.sh of
https://github.com/k2-fsa/sherpa-onnx: only the files required for forced
alignment are kept (the extra files of the upstream bundle, e.g. final.mdl,
phone_lm.fst, graphemes.txt, phone_pdf.counts, are dropped).

## Files

- meta.json      feature chain + silence/OOV parameters
- final.alimdl   alignment model (kaldi TransitionModel + AmDiagGmm)
- tree           context dependency tree
- lda.mat        LDA transform of the feature chain
- phones.txt     phone table
- lexicon.txt    pronunciation dictionary: one pronunciation per line,
                 "key phone1 phone2 ..."; multiple lines under one key =
                 multiple pronunciations (all kept as parallel paths; the
                 acoustics pick the best one). The words are enumerated
                 from lexicon-en-us.txt and looked up in this model's MFA
                 dictionary; words without a pronunciation are not
                 included and the aligner reports them as out-of-vocabulary
                 at runtime. Multi-word phrases keep their combined
                 readings; " || " separates their per-word phone groups.
- en.wav        test wav (its transcript is the --text below)

## License

License: CC BY 4.0 (as declared by the upstream model card). This model
follows whatever license the original model uses; see the
[source repository](https://github.com/MontrealCorpusTools/mfa-models) for
the license terms.

## Usage

The test wav en.wav is included so the package works out of the box.
sherpa-onnx-fa is only a demo: the same model can be used through the
APIs of all languages supported by sherpa-onnx.

  sherpa-onnx-fa \\
    --model-dir=$pkg \\
    --lexicon=$pkg/lexicon.txt \\
    --wav=$pkg/en.wav \\
    --text="ask not what your country can do for you, ask what you can do for your country." \\
    --out=output.TextGrid
EOF

ls -lh "$stage"

(cd package && tar cjf "../$pkg.tar.bz2" "$pkg")
rm -rf package
echo "wrote $pkg.tar.bz2"
