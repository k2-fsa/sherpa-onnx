#!/bin/bash
#
# Script to download and setup all required models for Sherpa-ONNX Combined WASM demo
#

set -e

# Create a tmp directory for downloads
mkdir -p tmp
cd tmp

echo "===== Setting up assets for Sherpa-ONNX Combined WASM Demo ====="
echo ""

# Function to check if a file exists and download only if needed
download_if_missing() {
  local target_file="../$1"
  local download_url="$2"
  local is_archive="$3"
  local extract_dir="$4"
  
  if [ -f "$target_file" ]; then
    echo "File $target_file already exists. Skipping download."
    return 0
  fi
  
  echo "Downloading $download_url..."
  
  if [ "$is_archive" = "yes" ]; then
    wget -q "$download_url"
    local file=$(basename "$download_url")
    
    echo "Extracting $file..."
    tar xvf "$file"
    rm "$file"
    
    if [ ! -z "$extract_dir" ]; then
      mv "$extract_dir" "../$(dirname "$target_file")"
    fi
  else
    wget -q -O "$target_file" "$download_url"
  fi
  
  echo "Downloaded and setup $target_file"
}

# Create subdirectories for each model type
mkdir -p ../asr ../vad ../tts ../speakers ../enhancement ../kws

echo "1. Setting up ASR Models (Speech Recognition)..."

# Download ASR models
wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
rm sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2

# Rename for compatibility
mv sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.int8.onnx ../asr/encoder.onnx
mv sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx ../asr/decoder.onnx
mv sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.int8.onnx ../asr/joiner.onnx
mv sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt ../asr/tokens.txt
rm -rf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/

echo "2. Setting up VAD Models (Voice Activity Detection)..."

# Download VAD model
wget -q -O ../vad/silero_vad.onnx https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

echo "3. Setting up TTS Models (Text-to-Speech)..."

# Download TTS models
wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
tar xvf vits-piper-en_US-amy-low.tar.bz2
rm vits-piper-en_US-amy-low.tar.bz2

mv vits-piper-en_US-amy-low/en_US-amy-low.onnx ../tts/model.onnx
mv vits-piper-en_US-amy-low/tokens.txt ../tts/tokens.txt
# Create a zip archive of the espeak-ng-data directory for efficient loading in WASM
mv vits-piper-en_US-amy-low/espeak-ng-data ../tts/

# Create zip archive of espeak-ng-data
echo "Creating zip archive of espeak-ng-data..."
cd ../tts
zip -r espeak-ng-data.zip espeak-ng-data/
cd ../../tmp

rm -rf vits-piper-en_US-amy-low/

echo "4. Setting up Speaker Diarization Models..."

# Download speaker diarization models
wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
tar xvf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
rm sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
mv sherpa-onnx-pyannote-segmentation-3-0/model.onnx ../speakers/segmentation.onnx
rm -rf sherpa-onnx-pyannote-segmentation-3-0

wget -q -O ../speakers/embedding.onnx https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx

echo "5. Setting up Speech Enhancement Models..."

# Download speech enhancement model
wget -q -O ../enhancement/gtcrn.onnx https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx

echo "6. Setting up Keyword Spotting Models..."

# Download keyword spotting models
wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
tar xvf sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
rm sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2

mv sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/encoder-epoch-12-avg-2-chunk-16-left-64.onnx ../kws/encoder.onnx
mv sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/decoder-epoch-12-avg-2-chunk-16-left-64.onnx ../kws/decoder.onnx
mv sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/joiner-epoch-12-avg-2-chunk-16-left-64.onnx ../kws/joiner.onnx
mv sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/tokens.txt ../kws/tokens.txt
rm -rf sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01

# Clean up tmp directory
cd ..
rm -rf tmp

echo ""
echo "===== All assets have been downloaded and set up successfully! ====="
echo ""
echo "To run the demo:"
echo "1. Build the WASM module: ./build-wasm-combined.sh"
echo "2. Start a local server: cd ../.. && python3 -m http.server 8080"
echo "3. Open your browser and go to: http://localhost:8080/wasm/combined/"
echo "" 