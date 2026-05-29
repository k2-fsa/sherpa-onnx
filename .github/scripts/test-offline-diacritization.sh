#!/usr/bin/env bash

set -ex

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

echo "EXE is $EXE"
echo "PATH: $PATH"

which $EXE

log "------------------------------------------------------------"
log "Download the CATT (Encoder-Only) diacritization model       "
log "------------------------------------------------------------"

tmp_dir=catt_eo_model_onnx
mkdir -p $tmp_dir
cd $tmp_dir
curl -SL -O https://github.com/abjadai/catt/releases/download/v2/eo_model_onnx.zip
unzip -o eo_model_onnx.zip
rm eo_model_onnx.zip
cd ..

assert_diacritized() {
  local input="$1"
  local output="$2"
  python - "$input" "$output" <<'PY'
import sys
inp, out = sys.argv[1], sys.argv[2]
assert out, "output is empty"
assert out != inp, "output equals input (no diacritization applied)"
# 0x064B fathatan, 0x0652 sukun, 0x0670 superscript alef
tashkeel = set(range(0x064B, 0x0653)) | {0x0670}
assert any(ord(c) in tashkeel for c in out), \
    "output contains no tashkeel codepoint"
print("OK")
PY
}

first_input="وقالت مجلة نيوزويك الأمريكية التحديث الجديد ل إنستجرام يمكن أن يساهم في إيقاف وكشف الحسابات المزورة بسهولة شديدة"
first_output=$($EXE \
 --debug=1 \
 --catt-encoder=$tmp_dir/encoder.onnx \
 --catt-decoder=$tmp_dir/decoder.onnx \
 "$first_input")
assert_diacritized "$first_input" "$first_output"

second_input="اللغة العربية من أقدم اللغات السامية"
second_output=$($EXE \
 --debug=1 \
 --catt-encoder=$tmp_dir/encoder.onnx \
 --catt-decoder=$tmp_dir/decoder.onnx \
 "$second_input")
assert_diacritized "$second_input" "$second_output"

rm -rf $tmp_dir
