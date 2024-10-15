#!/usr/bin/env bash
# Copyright      2024  Xiaomi Corp.        (authors: Fangjun Kuang)

set -ex

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# 36000 hours of English data
url=https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/parakeet-tdt_ctc-110m
name=$(basename $url)
doc="parakeet-tdt_ctc-110m is an ASR model that transcribes speech with Punctuations and Capitalizations of the English alphabet. It was trained on 36K hours of English speech collected and prepared by NVIDIA NeMo and Suno teams."

log "Process $name at $url"
./export-onnx-ctc-non-streaming.py --model $name --doc "$doc"
d=sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000
mkdir -p $d
mv -v model.onnx $d/
mv -v tokens.txt $d/
ls -lh $d

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
log "Download test data"
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/spoken-language-identification-test-wavs.tar.bz2
tar xvf spoken-language-identification-test-wavs.tar.bz2
rm spoken-language-identification-test-wavs.tar.bz2
data=spoken-language-identification-test-wavs

curl -SL -O https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav
mv 2086-149220-0033.wav en.wav

d=sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000
python3 ./test-onnx-ctc-non-streaming.py \
  --model $d/model.onnx \
  --tokens $d/tokens.txt \
  --wav $data/en-english.wav
mkdir -p $d/test_wavs

cp en.wav $d/test_wavs/0.wav
cp -v $data/en-english.wav $d/test_wavs/1.wav

d=sherpa-onnx-nemo-fast-conformer-ctc-en-24500
python3 ./test-onnx-ctc-non-streaming.py \
  --model $d/model.onnx \
  --tokens $d/tokens.txt \
  --wav $data/en-english.wav
mkdir -p $d/test_wavs
cp en.wav $d/test_wavs/0.wav
cp -v $data/en-english.wav $d/test_wavs

d=sherpa-onnx-nemo-fast-conformer-ctc-es-1424
python3 ./test-onnx-ctc-non-streaming.py \
  --model $d/model.onnx \
  --tokens $d/tokens.txt \
  --wav $data/es-spanish.wav
mkdir -p $d/test_wavs
cp -v $data/es-spanish.wav $d/test_wavs

d=sherpa-onnx-nemo-fast-conformer-ctc-en-de-es-fr-14288
mkdir -p $d/test_wavs
for w in en-english.wav de-german.wav es-spanish.wav fr-french.wav; do
  python3 ./test-onnx-ctc-non-streaming.py \
    --model $d/model.onnx \
    --tokens $d/tokens.txt \
    --wav $data/$w
  cp -v $data/$w $d/test_wavs
done

d=sherpa-onnx-nemo-fast-conformer-ctc-be-de-en-es-fr-hr-it-pl-ru-uk-20k
mkdir -p $d/test_wavs
for w in en-english.wav de-german.wav es-spanish.wav fr-french.wav hr-croatian.wav it-italian.wav po-polish.wav ru-russian.wav uk-ukrainian.wav; do
  python3 ./test-onnx-ctc-non-streaming.py \
    --model $d/model.onnx \
    --tokens $d/tokens.txt \
    --wav $data/$w
  cp -v $data/$w $d/test_wavs
done
