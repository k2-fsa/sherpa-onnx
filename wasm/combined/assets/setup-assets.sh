#!/bin/bash
#
# Script to download and setup all required models for Sherpa-ONNX Combined WASM demo
#

set -e

# Parse command line arguments
FORCE=false
for arg in "$@"
do
    case $arg in
        --force)
        FORCE=true
        shift
        ;;
    esac
done

echo "===== Setting up assets for Sherpa-ONNX Combined WASM Demo ====="
echo ""

if [ "$FORCE" = true ]; then
    echo "Force mode enabled - will delete existing assets"
fi

# Create subdirectories for each model type
mkdir -p asr vad tts speakers enhancement kws

# Function to check if a directory exists and has content
check_dir_not_empty() {
    local dir="$1"
    if [ -d "$dir" ] && [ "$(ls -A "$dir" 2>/dev/null)" ]; then
        return 0  # Directory exists and not empty
    else
        return 1  # Directory doesn't exist or is empty
    fi
}

# Create a tmp directory for downloads
mkdir -p tmp
cd tmp

# Download ASR models
echo "1. Setting up ASR Models (Speech Recognition)..."
if [ "$FORCE" = true ] || ! check_dir_not_empty "../asr"; then
    # Clean up if force is enabled
    if [ "$FORCE" = true ] && [ -d "../asr" ]; then
        rm -rf "../asr"
        mkdir -p "../asr"
    fi
    
    # Download and extract ASR models
    wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
    tar xvf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
    rm sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2

    # Rename for compatibility
    mv sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.int8.onnx ../asr/encoder.onnx
    mv sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx ../asr/decoder.onnx
    mv sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.int8.onnx ../asr/joiner.onnx
    mv sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt ../asr/tokens.txt
    rm -rf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/
    echo "ASR models downloaded and set up."
else
    echo "ASR models already exist. Skipping download. Use --force to re-download."
fi

# Download VAD model
echo "2. Setting up VAD Models (Voice Activity Detection)..."
if [ "$FORCE" = true ] || ! [ -f "../vad/silero_vad.onnx" ]; then
    # Clean up if force is enabled
    if [ "$FORCE" = true ] && [ -d "../vad" ]; then
        rm -rf "../vad"
        mkdir -p "../vad"
    fi
    
    wget -q -O ../vad/silero_vad.onnx https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
    echo "VAD model downloaded and set up."
else
    echo "VAD model already exists. Skipping download. Use --force to re-download."
fi

# Download TTS models
echo "3. Setting up TTS Models (Text-to-Speech)..."
if [ "$FORCE" = true ] || ! check_dir_not_empty "../tts"; then
    # Clean up if force is enabled
    if [ "$FORCE" = true ] && [ -d "../tts" ]; then
        rm -rf "../tts"
        mkdir -p "../tts"
    fi
    
    wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
    tar xvf vits-piper-en_US-amy-low.tar.bz2
    rm vits-piper-en_US-amy-low.tar.bz2

    # Move required files to TTS directory
    mv vits-piper-en_US-amy-low/en_US-amy-low.onnx ../tts/model.onnx
    mv vits-piper-en_US-amy-low/tokens.txt ../tts/tokens.txt
    
    # Handle espeak-ng-data directory safely
    if [ -d "../tts/espeak-ng-data" ] && [ "$FORCE" = false ]; then
        echo "espeak-ng-data directory already exists. Skipping..."
    else
        # Remove existing directory if it exists and we're forcing
        if [ -d "../tts/espeak-ng-data" ]; then
            rm -rf "../tts/espeak-ng-data"
        fi
        mv vits-piper-en_US-amy-low/espeak-ng-data ../tts/
    fi
    
    # Create zip archive of espeak-ng-data if needed
    if [ ! -f "../tts/espeak-ng-data.zip" ] || [ "$FORCE" = true ]; then
        echo "Creating zip archive of espeak-ng-data..."
        cd ../tts
        # Remove existing zip if force is enabled
        if [ -f "espeak-ng-data.zip" ] && [ "$FORCE" = true ]; then
            rm espeak-ng-data.zip
        fi
        zip -r espeak-ng-data.zip espeak-ng-data/
        cd ../tmp
    else
        echo "espeak-ng-data.zip already exists. Skipping..."
    fi
    
    rm -rf vits-piper-en_US-amy-low/
    echo "TTS models downloaded and set up."
else
    echo "TTS models already exist. Skipping download. Use --force to re-download."
fi

# Download speaker diarization models
echo "4. Setting up Speaker Diarization Models..."
if [ "$FORCE" = true ] || ! check_dir_not_empty "../speakers"; then
    # Clean up if force is enabled
    if [ "$FORCE" = true ] && [ -d "../speakers" ]; then
        rm -rf "../speakers"
        mkdir -p "../speakers"
    fi
    
    # Download segmentation model
    if [ "$FORCE" = true ] || ! [ -f "../speakers/segmentation.onnx" ]; then
        wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
        tar xvf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
        rm sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
        mv sherpa-onnx-pyannote-segmentation-3-0/model.onnx ../speakers/segmentation.onnx
        rm -rf sherpa-onnx-pyannote-segmentation-3-0
    fi
    
    # Download embedding model
    if [ "$FORCE" = true ] || ! [ -f "../speakers/embedding.onnx" ]; then
        wget -q -O ../speakers/embedding.onnx https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
    fi
    echo "Speaker diarization models downloaded and set up."
else
    echo "Speaker diarization models already exist. Skipping download. Use --force to re-download."
fi

# Download speech enhancement model
echo "5. Setting up Speech Enhancement Models..."
if [ "$FORCE" = true ] || ! [ -f "../enhancement/gtcrn.onnx" ]; then
    # Clean up if force is enabled
    if [ "$FORCE" = true ] && [ -d "../enhancement" ]; then
        rm -rf "../enhancement"
        mkdir -p "../enhancement"
    fi
    
    wget -q -O ../enhancement/gtcrn.onnx https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx
    echo "Speech enhancement model downloaded and set up."
else
    echo "Speech enhancement model already exists. Skipping download. Use --force to re-download."
fi

# Download keyword spotting models
echo "6. Setting up Keyword Spotting Models..."
if [ "$FORCE" = true ] || ! check_dir_not_empty "../kws"; then
    # Clean up if force is enabled
    if [ "$FORCE" = true ] && [ -d "../kws" ]; then
        rm -rf "../kws"
        mkdir -p "../kws"
    fi
    
    wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
    tar xvf sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
    rm sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2

    mv sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/encoder-epoch-12-avg-2-chunk-16-left-64.onnx ../kws/encoder.onnx
    mv sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/decoder-epoch-12-avg-2-chunk-16-left-64.onnx ../kws/decoder.onnx
    mv sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/joiner-epoch-12-avg-2-chunk-16-left-64.onnx ../kws/joiner.onnx
    mv sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/tokens.txt ../kws/tokens.txt
    rm -rf sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01
    echo "Keyword spotting models downloaded and set up."
else
    echo "Keyword spotting models already exist. Skipping download. Use --force to re-download."
fi

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