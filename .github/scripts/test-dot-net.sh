#!/usr/bin/env bash

cd dotnet-examples/

cd spoken-language-identification
./run.sh

cd ../online-decode-files
./run-zipformer2-ctc.sh
./run-transducer.sh
./run-paraformer.sh

cd ../offline-decode-files
./run-nemo-ctc.sh
./run-paraformer.sh
./run-zipformer.sh
./run-hotwords.sh
./run-whisper.sh
./run-tdnn-yesno.sh

cd ../offline-tts
./run-aishell3.sh
./run-piper.sh
ls -lh

cd ../..

mkdir tts

cp dotnet-examples/offline-tts/*.wav ./tts
