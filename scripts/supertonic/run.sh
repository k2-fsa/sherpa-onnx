#!/bin/bash
# Copyright (c)  2026 zengyw

set -e

# Download helper.py from GitHub
HELPER_URL="https://raw.githubusercontent.com/supertone-inc/supertonic/main/py/helper.py"
HELPER_PATH="helper.py"

if [ ! -f "$HELPER_PATH" ]; then
    echo "Downloading helper.py from GitHub..."
    curl -L -o "$HELPER_PATH" "$HELPER_URL" || wget -O "$HELPER_PATH" "$HELPER_URL"
    if [ $? -eq 0 ]; then
        echo "Successfully downloaded helper.py"
    else
        echo "Error: Failed to download helper.py"
        exit 1
    fi
fi

# Stage control: run from stage N to stop_stage M
# Usage: ./run.sh [STAGE] [STOP_STAGE]
# Default: run all stages (stage=0, stop-stage=4)

usage() {
    echo "Usage: $0 [STAGE] [STOP_STAGE]"
    echo ""
    echo "Arguments:"
    echo "  STAGE       Start stage (default: 0)"
    echo "  STOP_STAGE  Stop stage (default: 4)"
    echo ""
    echo "Stages:"
    echo "  0: Download ONNX models (if needed)"
    echo "  1: Generate calibration configs"
    echo "  2: Dump calibration data"
    echo "  3: Quantize ONNX models to INT8"
    echo "  4: Generate voice.bin (from assets/voice_styles/*.json)"
    echo ""
    echo "Examples:"
    echo "  ./run.sh              # Run all stages (0-4)"
    echo "  ./run.sh 1 2         # Run stages 1-2 only"
    echo "  ./run.sh 3            # Only run stage 3"
    echo "  ./run.sh 0 0         # Only download models"
    echo "  ./run.sh 4            # Only generate voice.bin"
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

STAGE=${1:-0}
STOP_STAGE=${2:-4}

echo "========================================"
echo "  SuperTonic INT8 Quantization Pipeline"
echo "========================================"
echo "Stage: $STAGE -> $STOP_STAGE"
echo ""

# Stage 0: Download ONNX models if not exists
if [ ${STAGE} -le 0 ] && [ ${STOP_STAGE} -ge 0 ]; then
    echo ""
    echo "Stage 0: Check/Download ONNX Models"

    if [ ! -d "./assets/onnx" ] || [ -z "$(ls -A ./assets/onnx 2>/dev/null)" ]; then
        echo "ONNX models not found, downloading..."
        mkdir -p ./assets
        modelscope download --model Supertone/supertonic-2 --local_dir ./assets
        echo "Download completed"
    else
        echo "ONNX models already exist, skipping download"
    fi
    echo "Stage 0 Completed"
    echo ""
fi

# Stage 1: Generate calibration configs
if [ ${STAGE} -le 1 ] && [ ${STOP_STAGE} -ge 1 ]; then
    echo ""
    echo "Stage 1: Generate Calibration Configs"
    python gen_calib_configs.py
    echo "Stage 1 Completed"
    echo ""
fi

# Stage 2: Dump calibration data
if [ ${STAGE} -le 2 ] && [ ${STOP_STAGE} -ge 2 ]; then
    echo ""
    echo "Stage 2: Dump Calibration Data"
    python dump_inputs.py --config-file calib_configs.json --clear
    echo "Stage 2 Completed"
    echo ""
fi

# Stage 3: Quantize ONNX models to INT8
if [ ${STAGE} -le 3 ] && [ ${STOP_STAGE} -ge 3 ]; then
    echo ""
    echo "Stage 3: Quantize ONNX to INT8"
    python3 convert.py \
      --src-dir ./assets/onnx \
      --dst-dir ./onnx_int8 \
      --calib-dir ./calib \
      --preprocess ort \
      --vocoder-calib-from-ve \
      --exclude-last-conv 8 \
      --ve-calib-limit 100 \
      --vocoder-calib-limit 100 \
      --pad-percentile 90
    echo "Stage 3 Completed"
    echo ""
fi

# Stage 4: Generate voice.bin (merge all JSONs in assets/voice_styles)
if [ ${STAGE} -le 4 ] && [ ${STOP_STAGE} -ge 4 ]; then
    echo ""
    echo "Stage 4: Generate voice.bin"
    python3 generate_voices_bin.py
    echo "Stage 4 Completed"
    echo ""
fi

echo "========================================"
echo "  Pipeline Completed!"
echo "========================================"
