# Whisper Timestamp Accuracy Benchmark

This directory contains tools for benchmarking sherpa-onnx Whisper word timestamp accuracy against ground truth alignments from the Montreal Forced Aligner (MFA).

## Overview

The benchmark suite evaluates how accurately sherpa-onnx predicts word-level timestamps by comparing against MFA alignments on LibriSpeech data. MFA provides high-quality forced alignments that serve as ground truth for measuring timestamp accuracy.

## Scripts

### `download_librispeech_test_data.py`

Downloads and prepares the benchmark dataset:
- LibriSpeech dev-clean audio (converted to 16kHz mono WAV)
- MFA word alignments with precise word boundaries

**Usage:**
```bash
uv run python scripts/benchmark/download_librispeech_test_data.py [--num-utterances 200]
```

**Options:**
- `--num-utterances` - Number of utterances to include (default: 200)
- `--output-dir` - Output directory (default: `benchmark_data`)
- `--skip-download` - Skip download step and use existing files

**Output:**
- `benchmark_data/audio/*.wav` - Audio files
- `benchmark_data/manifest.json` - Mapping of audio files to ground truth timestamps

**Requirements:**
- `gdown` (for Google Drive downloads)
- `ffmpeg` or `sox` (for audio conversion)

### `run_timestamp_benchmark.py`

Runs the timestamp accuracy benchmark against the downloaded ground truth.

**Usage:**
```bash
PYTHONPATH=build/lib:sherpa-onnx/python uv run python scripts/benchmark/run_timestamp_benchmark.py \
    --encoder ./whisper-tiny-attention/tiny-encoder.onnx \
    --decoder ./whisper-tiny-attention/tiny-decoder.onnx \
    --tokens ./whisper-tiny-attention/tiny-tokens.txt
```

**Options:**
- `--encoder` - Path to Whisper encoder ONNX model (required)
- `--decoder` - Path to Whisper decoder ONNX model (required)
- `--tokens` - Path to tokens file (required)
- `--data-dir` - Directory with manifest and audio (default: `benchmark_data`)
- `--output-dir` - Output directory for results (default: `benchmark_results`)
- `--language` - Language code (default: `en`)
- `--num-workers` - Number of parallel workers (default: 1)

**Parallel Processing:**
```bash
# Run with 4 workers for faster benchmarking
PYTHONPATH=build/lib:sherpa-onnx/python uv run python scripts/benchmark/run_timestamp_benchmark.py \
    --encoder ./whisper-tiny-attention/tiny-encoder.onnx \
    --decoder ./whisper-tiny-attention/tiny-decoder.onnx \
    --tokens ./whisper-tiny-attention/tiny-tokens.txt \
    --num-workers 4
```

Note: Each worker loads its own model copy, so memory usage scales linearly with worker count.

**Requirements:**
- `numpy`
- `jiwer` (for WER calculation)
- Built sherpa-onnx library

**Note on PYTHONPATH:** This script uses `PYTHONPATH=build/lib:sherpa-onnx/python` instead of `pip install sherpa-onnx` to allow rapid iteration when developing C++ code. After running `make` in the build directory, you can immediately test without reinstalling the package.

## Output Format

### `details_YYYYMMDD_HHMMSS.csv`

Per-word timing errors with columns:
- `utterance_id` - Utterance identifier
- `word_index` - Word position in utterance
- `word` - The word text
- `gt_start`, `gt_end` - Ground truth timestamps (seconds)
- `pred_start`, `pred_end` - Predicted timestamps (seconds)
- `matched` - Whether the word was successfully aligned
- `start_error_ms`, `end_error_ms` - Timing errors in milliseconds

### `summary_YYYYMMDD_HHMMSS.csv`

Per-utterance aggregate statistics:
- `utterance_id` - Utterance identifier
- `num_gt_words`, `num_pred_words`, `num_matched` - Word counts
- `match_rate` - Fraction of ground truth words matched
- `wer` - Word Error Rate
- `mean_start_error_ms`, `median_start_error_ms`, `max_start_error_ms` - Start time error statistics
- `mean_end_error_ms`, `median_end_error_ms`, `max_end_error_ms` - End time error statistics
- `pct_within_20ms`, `pct_within_50ms` - Percentage of words within accuracy thresholds

## Metrics Explained

- **Start/End Time Error**: Absolute difference between predicted and ground truth timestamps
- **Match Rate**: How many ground truth words were successfully aligned with predictions
- **WER (Word Error Rate)**: Standard ASR accuracy metric (lower is better)
- **Accuracy Thresholds**: Percentage of words with start time error within 20ms, 50ms, or 100ms

## Example Workflow

```bash
# 1. Build sherpa-onnx
cd build && make -j8 && cd ..

# 2. Export a Whisper model with attention outputs
uv run python scripts/whisper/export-onnx.py --model tiny --with-attention --output-dir ./whisper-tiny-attention

# 3. Download benchmark data
uv run python scripts/benchmark/download_librispeech_test_data.py --num-utterances 200

# 4. Run the benchmark
PYTHONPATH=build/lib:sherpa-onnx/python uv run python scripts/benchmark/run_timestamp_benchmark.py \
    --encoder ./whisper-tiny-attention/tiny-encoder.onnx \
    --decoder ./whisper-tiny-attention/tiny-decoder.onnx \
    --tokens ./whisper-tiny-attention/tiny-tokens.txt \
    --num-workers 4

# 5. Review results in benchmark_results/
```

## Data Sources and Citations

### LibriSpeech Corpus

The audio data comes from the [LibriSpeech](https://www.openslr.org/12/) ASR corpus:

> Panayotov, V., Chen, G., Povey, D., & Khudanpur, S. (2015). LibriSpeech: An ASR corpus based on public domain audio books. In *2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)* (pp. 5206-5210). IEEE. https://doi.org/10.1109/ICASSP.2015.7178964

LibriSpeech is derived from read audiobooks from the [LibriVox](https://librivox.org/) project and is freely available under a CC BY 4.0 license.

### Montreal Forced Aligner (MFA)

The ground truth word alignments were generated using the [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/):

> McAuliffe, M., Socolof, M., Mihuc, S., Wagner, M., & Sonderegger, M. (2017). Montreal Forced Aligner: Trainable text-speech alignment using Kaldi. In *Proceedings of Interspeech 2017* (pp. 498-502). https://doi.org/10.21437/Interspeech.2017-1386

MFA is an open-source forced alignment tool that uses Kaldi for acoustic modeling.

### Pre-computed LibriSpeech Alignments

The pre-computed MFA alignments for LibriSpeech are provided by the [librispeech-alignments](https://github.com/CorentinJ/librispeech-alignments) project by Corentin Jemine.

## License

The LibriSpeech corpus is released under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license. Please ensure compliance with all applicable licenses when using this benchmark data.
