#!/usr/bin/env bash
set -euo pipefail

# Download the large_pinyin MFA acoustic model and generate lexicon.txt
# (hanzi/phrase -> phones). See README.md for details.

cd "$(dirname "$0")"

url=https://github.com/lars76/forced-alignment-chinese/releases/download/model_files/large_pinyin.zip
zip=download/large_pinyin.zip

mkdir -p download
if [ ! -f download/.model_extracted ]; then
  if [ ! -f "$zip" ]; then
    echo "Downloading $url"
    curl -SL --fail -o "$zip" "$url"
  fi
  unzip -q -o "$zip" -d download
  # the release bundle contains the acoustic model as a nested *_acoustic.zip
  for z in download/*_acoustic.zip; do
    unzip -q -o "$z" -d download
  done
  touch download/.model_extracted
fi

# Python dependencies of generate_lexicon.py
python3 -c "import pypinyin" 2>/dev/null || python3 -m pip install pypinyin
python3 -c "import pinyin_to_ipa" 2>/dev/null || python3 -m pip install pinyin_to_ipa

python3 -I generate_lexicon.py
