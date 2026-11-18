#!/usr/bin/env bash
set -ex
cd go-api-examples/forced-alignment-chinese
go mod tidy
go build
./run.sh

grep -q 'name = "phones"' zh.TextGrid
grep -q 'name = "words"' zh.TextGrid
cat zh.TextGrid

cd ../..
mkdir -p test-forced-alignment-chinese
cp go-api-examples/forced-alignment-chinese/zh-aishell3-pinyin/zh.wav test-forced-alignment-chinese/
mv go-api-examples/forced-alignment-chinese/zh.TextGrid test-forced-alignment-chinese/

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
    --wav test-forced-alignment-chinese/zh.wav \
    --textgrid test-forced-alignment-chinese/zh.TextGrid \
    --out-dir test-forced-alignment-chinese/word-clips-zh
else
  echo "Python is not available; skipping the word clips"
fi

rm -rf go-api-examples/forced-alignment-chinese/zh-aishell3-pinyin
