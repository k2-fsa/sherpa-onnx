#!/usr/bin/env bash
set -ex
cd go-api-examples/forced-alignment-english
go mod tidy
go build
./run.sh

grep -q 'name = "phones"' en.TextGrid
grep -q 'name = "words"' en.TextGrid
cat en.TextGrid

cd ../..
mkdir -p test-forced-alignment-english
cp go-api-examples/forced-alignment-english/english-mfa-v3.1.0/en.wav test-forced-alignment-english/
mv go-api-examples/forced-alignment-english/en.TextGrid test-forced-alignment-english/

# The word clips need Python; skip them where it is missing (e.g. Termux).
# The helper itself skips when soundfile is not installed. Termux calls the
# interpreter `python`, most other platforms `python3`.
if command -v python3 >/dev/null 2>&1; then
  py=python3
elif command -v python >/dev/null 2>&1; then
  py=python
else
  py=
fi

if [ -n "$py" ]; then
  $py python-api-examples/textgrid-to-word-wavs.py \
    --wav test-forced-alignment-english/en.wav \
    --textgrid test-forced-alignment-english/en.TextGrid \
    --out-dir test-forced-alignment-english/word-clips-en
else
  echo "Python is not available; skipping the word clips"
fi

rm -rf go-api-examples/forced-alignment-english/english-mfa-v3.1.0
