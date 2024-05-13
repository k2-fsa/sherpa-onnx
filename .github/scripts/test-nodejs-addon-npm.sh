#!/usr/bin/env bash

set -ex

d=nodejs-addon-examples
echo "dir: $d"
cd $d

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
rm sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2

node test_asr_streaming_transducer.js

rm -rf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
rm sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2

node ./test_asr_streaming_ctc.js

# To decode with HLG.fst
node ./test_asr_streaming_ctc_hlg.js
rm -rf sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
tar xvf sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
rm sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2

node ./test_asr_streaming_paraformer.js
rm -rf sherpa-onnx-streaming-paraformer-bilingual-zh-en

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-en-2023-04-01.tar.bz2
tar xvf sherpa-onnx-zipformer-en-2023-04-01.tar.bz2
rm sherpa-onnx-zipformer-en-2023-04-01.tar.bz2

node ./test_asr_non_streaming_transducer.js
rm -rf sherpa-onnx-zipformer-en-2023-04-01

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2
tar xvf sherpa-onnx-whisper-tiny.en.tar.bz2
rm sherpa-onnx-whisper-tiny.en.tar.bz2

node ./test_asr_non_streaming_whisper.js
rm -rf sherpa-onnx-whisper-tiny.en

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-fast-conformer-ctc-be-de-en-es-fr-hr-it-pl-ru-uk-20k.tar.bz2
tar xvf sherpa-onnx-nemo-fast-conformer-ctc-be-de-en-es-fr-hr-it-pl-ru-uk-20k.tar.bz2
rm sherpa-onnx-nemo-fast-conformer-ctc-be-de-en-es-fr-hr-it-pl-ru-uk-20k.tar.bz2

node ./test_asr_non_streaming_nemo_ctc.js
rm -rf sherpa-onnx-nemo-fast-conformer-ctc-be-de-en-es-fr-hr-it-pl-ru-uk-20k

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-03-28.tar.bz2
tar xvf sherpa-onnx-paraformer-zh-2023-03-28.tar.bz2
rm sherpa-onnx-paraformer-zh-2023-03-28.tar.bz2

node ./test_asr_non_streaming_paraformer.js
rm -rf sherpa-onnx-paraformer-zh-2023-03-28
