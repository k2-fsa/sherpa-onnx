#!/usr/bin/env bash
# Copyright      2024  Xiaomi Corp.        (authors: Fangjun Kuang)

set -ex

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# 8500 hours of English speech
url=https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_fastconformer_hybrid_large_pc
name=$(basename $url)
doc="This collection contains the English FastConformer Hybrid (Transducer and CTC) Large model (around 114M parameters) with Punctuation and Capitalization on NeMo ASRSet En PC with around 8500 hours of English speech (SPGI 1k, VoxPopuli, MCV11, Europarl-ASR, Fisher, LibriSpeech, NSC1, MLS). It utilizes a Google SentencePiece [1] tokenizer with a vocabulary size of 1024. It transcribes text in upper and lower case English alphabet along with spaces, periods, commas, question marks, and a few other characters."

log "Process $name at $url"
./export-onnx-ctc-non-streaming.py --model $name --doc "$doc"

d=sherpa-onnx-nemo-fast-conformer-ctc-en-24500
mkdir -p $d
mv -v model.onnx $d/
mv -v tokens.txt $d/
ls -lh $d

url=https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_es_fastconformer_hybrid_large_pc
name=$(basename $url)
doc="This collection contains the Spanish FastConformer Hybrid (CTC and Transducer) Large model (around 114M parameters) with Punctuation and Capitalization. It is trained on the NeMo PnC ES ASRSET (Fisher, MCV12, MLS, Voxpopuli) containing 1424 hours of Spanish speech. It utilizes a Google SentencePiece [1] tokenizer with vocabulary size 1024, and transcribes text in upper and lower case Spanish alphabet along with spaces, period, comma, question mark and inverted question mark."

./export-onnx-ctc-non-streaming.py --model $name --doc "$doc"

d=sherpa-onnx-nemo-fast-conformer-ctc-es-1424
mkdir -p $d
mv -v model.onnx $d/
mv -v tokens.txt $d/
ls -lh $d

url=https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_multilingual_fastconformer_hybrid_large_pc_blend_eu
name=$(basename $url)
doc="This collection contains the Multilingual FastConformer Hybrid (Transducer and CTC) Large model (around 114M parameters) with Punctuation and Capitalization. It is trained on the NeMo PnC German, English, Spanish, and French ASR sets that contain 14,288 hours of speech in total. It utilizes a Google SentencePiece [1] tokenizer with vocabulary size 256 per language and transcribes text in upper and lower case along with spaces, periods, commas, question marks and a few other language-specific characters. The total tokenizer size is 2560, of which 1024 tokens are allocated to English, German, French, and Spanish. The remaining tokens are reserved for future languages."

./export-onnx-ctc-non-streaming.py --model $name --doc "$doc"

d=sherpa-onnx-nemo-fast-conformer-ctc-en-de-es-fr-14288
mkdir -p $d
mv -v model.onnx $d/
mv -v tokens.txt $d/
ls -lh $d

url=https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_multilingual_fastconformer_hybrid_large_pc
name=$(basename $url)
doc="This collection contains the Multilingual FastConformer Hybrid (Transducer and CTC) Large model (around 114M parameters) with Punctuation and Capitalization. It is trained on the NeMo PnC Belarusian, German, English, Spanish, French, Croatian, Italian, Polish, Russian, and Ukrainian ASR sets that contain ~20,000 hours of speech in total. It utilizes a Google SentencePiece [1] tokenizer with vocabulary size 256 per language (2560 total), and transcribes text in upper and lower case along with spaces, periods, commas, question marks and a few other language-specific characters."

./export-onnx-ctc-non-streaming.py --model $name --doc "$doc"

d=sherpa-onnx-nemo-fast-conformer-ctc-be-de-en-es-fr-hr-it-pl-ru-uk-20k
mkdir -p $d
mv -v model.onnx $d/
mv -v tokens.txt $d/
ls -lh $d

# Now test the exported model
