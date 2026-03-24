#!/usr/bin/env bash
set -ex

if [ ! -d ./vits-piper-de_DE-glados-high ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-de_DE-glados-high.tar.bz2
  tar xf vits-piper-de_DE-glados-high.tar.bz2
  rm vits-piper-de_DE-glados-high.tar.bz2
fi

cargo run --example vits_tts --   --model ./vits-piper-de_DE-glados-high/de_DE-glados-high.onnx   --tokens ./vits-piper-de_DE-glados-high/tokens.txt   --data-dir ./vits-piper-de_DE-glados-high/espeak-ng-data   --output ./generated-vits-de-rust.wav   --text "Alles hat ein Ende, nur die Wurst hat zwei."
