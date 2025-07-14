#!/usr/bin/env bash
# Copyright      2025  Xiaomi Corp.        (authors: Fangjun Kuang)

set -ex

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# 2200 hours of Portuguese speech
url=https://huggingface.co/nvidia/stt_pt_fastconformer_hybrid_large_pc
name=$(basename $url)
name="nvidia/$name"
doc="STT PT FastConformer Hybrid Transducer-CTC Large transcribes text in upper and lower case Portuguese alphabet along with spaces, period, comma, question mark. This collection contains the Brazilian Portuguese FastConformer Hybrid (Transducer and CTC) Large model (around 115M parameters) with punctuation and capitalization trained on around 2200h hours of Portuguese speech. "

log "Process $name at $url"
./export-onnx-ctc-non-streaming.py --model $name --doc "$doc"
d=sherpa-onnx-nemo-stt_pt_fastconformer_hybrid_large_pc
mkdir -p $d
mv -v model.onnx $d/
cp -v tokens.txt $d/
ls -lh $d

mkdir test_wavs
pushd test_wavs
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/pt_br.wav
popd
cp -a test_wavs $d

d=sherpa-onnx-nemo-stt_pt_fastconformer_hybrid_large_pc-int8
mkdir -p $d
mv -v model.int8.onnx $d/
mv -v tokens.txt $d/
ls -lh $d
mv test_wavs $d

python3 ./test-onnx-ctc-non-streaming.py \
  --model $d/model.int8.onnx \
  --tokens $d/tokens.txt \
  --wav $d/test_wavs/pt_br.wav


# 2500 hours of German speech
url=https://huggingface.co/nvidia/stt_de_fastconformer_hybrid_large_pc
name=$(basename $url)
name="nvidia/$name"
doc="This model transcribes speech in upper and lower case German alphabet along with spaces, periods, commas, and question marks. It is a 'large' version of FastConformer Transducer-CTC (around 115M parameters) model. This is a hybrid model trained on two losses: Transducer (default) and CTC."

log "Process $name at $url"
./export-onnx-ctc-non-streaming.py --model $name --doc "$doc"
d=sherpa-onnx-nemo-stt_de_fastconformer_hybrid_large_pc
mkdir -p $d
mv -v model.onnx $d/
cp -v tokens.txt $d/
ls -lh $d

mkdir test_wavs
pushd test_wavs
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/de.wav
popd
cp -a test_wavs $d

d=sherpa-onnx-nemo-stt_de_fastconformer_hybrid_large_pc-int8
mkdir -p $d
mv -v model.int8.onnx $d/
mv -v tokens.txt $d/
ls -lh $d
mv test_wavs $d

python3 ./test-onnx-ctc-non-streaming.py \
  --model $d/model.int8.onnx \
  --tokens $d/tokens.txt \
  --wav $d/test_wavs/de.wav
