#!/usr/bin/env bash

cd dotnet-examples/

cd ./offline-decode-files
./run-paraformer-itn.sh
./run-telespeech-ctc.sh
./run-nemo-ctc.sh
./run-paraformer.sh
./run-zipformer.sh
./run-hotwords.sh
./run-whisper.sh
./run-tdnn-yesno.sh

cd ../vad-non-streaming-asr-paraformer
./run.sh

cd ../offline-punctuation
./run.sh

cd ../speaker-identification
./run.sh

cd ../streaming-hlg-decoding/
./run.sh

cd ../spoken-language-identification
./run.sh

cd ../online-decode-files
./run-zipformer2-ctc.sh
./run-transducer.sh
./run-paraformer.sh

cd ../offline-tts
./run-aishell3.sh
./run-piper.sh
./run-hf-fanchen.sh
ls -lh

cd ../..

mkdir tts

cp dotnet-examples/offline-tts/*.wav ./tts
