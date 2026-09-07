#!/usr/bin/env bash
set -ex
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd go-api-examples/non-streaming-decode-files
go mod tidy
. "$SCRIPT_DIR/replace-sherpa-onnx-go.sh"
go build
./run-dolphin-ctc-base.sh
./run-fire-red-asr.sh
./run-moonshine.sh
./run-nemo-ctc.sh
./run-paraformer-itn.sh
./run-paraformer.sh
./run-sense-voice-small-with-hr.sh
./run-sense-voice-small.sh
./run-tdnn-yesno.sh
./run-telespeech-ctc.sh
./run-transducer.sh
./run-wenet-ctc.sh
./run-whisper.sh
./run-zipformer-ctc.sh
