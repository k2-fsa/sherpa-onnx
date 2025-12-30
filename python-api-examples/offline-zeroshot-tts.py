#!/usr/bin/env python3
#
# Copyright (c)  2025  Xiaomi Corporation

"""
This file demonstrates how to use sherpa-onnx Python API to generate audio
from text with prompt, i.e., zero shot text-to-speech.

Usage:

Example (zipvoice)

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2
tar xf sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos_24khz.onnx

python3 ./python-api-examples/offline-zeroshot-tts.py \
  --zipvoice-encoder sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/encoder.int8.onnx \
  --zipvoice-decoder sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/decoder.int8.onnx \
  --zipvoice-data-dir sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/espeak-ng-data \
  --zipvoice-lexicon sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/lexicon.txt \
  --zipvoice-tokens sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/tokens.txt \
  --zipvoice-vocoder vocos_24khz.onnx \
  --prompt-audio sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/test_wavs/leijun-1.wav \
  --zipvoice-num-steps 4 \
  --num-threads 4 \
  --prompt-text "那还是三十六年前, 一九八七年. 我呢考上了武汉大学的计算机系." \
  "小米的价值观是真诚, 热爱. 真诚，就是不欺人也不自欺. 热爱, 就是全心投入并享受其中."
"""

import argparse
import time
import wave
import numpy as np

from typing import Tuple

import sherpa_onnx
import soundfile as sf


def add_zipvoice_args(parser):
    parser.add_argument(
        "--zipvoice-tokens",
        type=str,
        default="",
        help="Path to tokens.txt for Zipvoice models.",
    )

    parser.add_argument(
        "--zipvoice-encoder",
        type=str,
        default="",
        help="Path to zipvoice text encoder model.",
    )

    parser.add_argument(
        "--zipvoice-decoder",
        type=str,
        default="",
        help="Path to zipvoice flow matching decoder model.",
    )

    parser.add_argument(
        "--zipvoice-data-dir",
        type=str,
        default="",
        help="Path to the dict directory of espeak-ng.",
    )

    parser.add_argument(
        "--zipvoice-lexicon",
        type=str,
        default="",
        help="Path to the lexicon.txt",
    )

    parser.add_argument(
        "--zipvoice-vocoder",
        type=str,
        default="",
        help="Path to the vocos vocoder.",
    )

    parser.add_argument(
        "--zipvoice-num-steps",
        type=int,
        default=4,
        help="Number of steps for Zipvoice.",
    )

    parser.add_argument(
        "--zipvoice-feat-scale",
        type=float,
        default=0.1,
        help="Scale factor for Zipvoice features.",
    )

    parser.add_argument(
        "--zipvoice-t-shift",
        type=float,
        default=0.5,
        help="Shift t to smaller ones if t-shift < 1.0.",
    )

    parser.add_argument(
        "--zipvoice-target-rms",
        type=float,
        default=0.1,
        help="Target speech normalization RMS value for Zipvoice.",
    )

    parser.add_argument(
        "--zipvoice-guidance-scale",
        type=float,
        default=1.0,
        help="The scale of classifier-free guidance during inference for for Zipvoice.",
    )


def read_wave(wave_filename: str) -> Tuple[np.ndarray, int]:
    """
    Args:
      wave_filename:
        Path to a wave file. It should be single channel and each sample should
        be 16-bit. Its sample rate does not need to be 16kHz.
    Returns:
      Return a tuple containing:
       - A 1-D array of dtype np.float32 containing the samples, which are
       normalized to the range [-1, 1].
       - sample rate of the wave file
    """

    with wave.open(wave_filename) as f:
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()  # it is in bytes
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32)

        samples_float32 = samples_float32 / 32768
        return samples_float32, f.getframerate()


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    add_zipvoice_args(parser)

    parser.add_argument(
        "--tts-rule-fsts",
        type=str,
        default="",
        help="Path to rule.fst",
    )

    parser.add_argument(
        "--max-num-sentences",
        type=int,
        default=1,
        help="""Max number of sentences in a batch to avoid OOM if the input
        text is very long. Set it to -1 to process all the sentences in a
        single batch. A smaller value does not mean it is slower compared
        to a larger one on CPU.
        """,
    )

    parser.add_argument(
        "--output-filename",
        type=str,
        default="./generated.wav",
        help="Path to save generated wave",
    )

    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="True to show debug messages",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        help="valid values: cpu, cuda, coreml",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads for neural network computation",
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed. Larger->faster; smaller->slower",
    )

    parser.add_argument(
        "--prompt-text",
        type=str,
        required=True,
        help="The transcription of prompt audio (Zipvoice)",
    )

    parser.add_argument(
        "--prompt-audio",
        type=str,
        required=True,
        help="The path to prompt audio (Zipvoice).",
    )

    parser.add_argument(
        "text",
        type=str,
        help="The input text to generate audio for",
    )

    return parser.parse_args()


def main():
    args = get_args()
    print(args)

    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            zipvoice=sherpa_onnx.OfflineTtsZipvoiceModelConfig(
                tokens=args.zipvoice_tokens,
                encoder=args.zipvoice_encoder,
                decoder=args.zipvoice_decoder,
                data_dir=args.zipvoice_data_dir,
                lexicon=args.zipvoice_lexicon,
                vocoder=args.zipvoice_vocoder,
                feat_scale=args.zipvoice_feat_scale,
                t_shift=args.zipvoice_t_shift,
                target_rms=args.zipvoice_target_rms,
                guidance_scale=args.zipvoice_guidance_scale,
            ),
            provider=args.provider,
            debug=args.debug,
            num_threads=args.num_threads,
        ),
        rule_fsts=args.tts_rule_fsts,
        max_num_sentences=args.max_num_sentences,
    )
    if not tts_config.validate():
        raise ValueError("Please check your config")

    tts = sherpa_onnx.OfflineTts(tts_config)

    start = time.time()
    prompt_samples, sample_rate = read_wave(args.prompt_audio)
    audio = tts.generate(
        args.text,
        args.prompt_text,
        prompt_samples,
        sample_rate,
        speed=args.speed,
        num_steps=args.zipvoice_num_steps,
    )
    end = time.time()

    if len(audio.samples) == 0:
        print("Error in generating audios. Please read previous error messages.")
        return

    elapsed_seconds = end - start
    audio_duration = len(audio.samples) / audio.sample_rate
    real_time_factor = elapsed_seconds / audio_duration

    sf.write(
        args.output_filename,
        audio.samples,
        samplerate=audio.sample_rate,
        subtype="PCM_16",
    )
    print(f"Saved to {args.output_filename}")
    print(f"The text is '{args.text}'")
    print(f"Elapsed seconds: {elapsed_seconds:.3f}")
    print(f"Audio duration in seconds: {audio_duration:.3f}")
    print(f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}")


if __name__ == "__main__":
    main()
