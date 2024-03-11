#!/usr/bin/env python3
#
# Copyright (c)  2023  Xiaomi Corporation

"""
This file demonstrates how to use sherpa-onnx Python API to generate audio
from text, i.e., text-to-speech.

Different from ./offline-tts.py, this file plays back the generated audio
while the model is still generating.

Usage:

Example (1/2)

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
tar xf vits-piper-en_US-amy-low.tar.bz2

python3 ./python-api-examples/offline-tts-play.py \
 --vits-model=./vits-piper-en_US-amy-low/en_US-amy-low.onnx \
 --vits-tokens=./vits-piper-en_US-amy-low/tokens.txt \
 --vits-data-dir=./vits-piper-en_US-amy-low/espeak-ng-data \
 --output-filename=./generated.wav \
 "Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar."

Example (2/2)

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-zh-aishell3.tar.bz2
tar xvf vits-zh-aishell3.tar.bz2

python3 ./python-api-examples/offline-tts-play.py \
 --vits-model=./vits-aishell3.onnx \
 --vits-lexicon=./lexicon.txt \
 --vits-tokens=./tokens.txt \
 --tts-rule-fsts=./rule.fst \
 --sid=21 \
 --output-filename=./liubei-21.wav \
 "勿以恶小而为之，勿以善小而不为。惟贤惟德，能服于人。122334"

You can find more models at
https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models

Please see
https://k2-fsa.github.io/sherpa/onnx/tts/index.html
for details.
"""

import argparse
import logging
import queue
import sys
import threading
import time

import numpy as np
import sherpa_onnx
import soundfile as sf

try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice first. You can use")
    print()
    print("  pip install sounddevice")
    print()
    print("to install it")
    sys.exit(-1)


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--vits-model",
        type=str,
        help="Path to vits model.onnx",
    )

    parser.add_argument(
        "--vits-lexicon",
        type=str,
        default="",
        help="Path to lexicon.txt",
    )

    parser.add_argument(
        "--vits-tokens",
        type=str,
        default="",
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "--vits-data-dir",
        type=str,
        default="",
        help="""Path to the dict director of espeak-ng. If it is specified,
        --vits-lexicon and --vits-tokens are ignored""",
    )

    parser.add_argument(
        "--tts-rule-fsts",
        type=str,
        default="",
        help="Path to rule.fst",
    )

    parser.add_argument(
        "--output-filename",
        type=str,
        default="./generated.wav",
        help="Path to save generated wave",
    )

    parser.add_argument(
        "--sid",
        type=int,
        default=0,
        help="""Speaker ID. Used only for multi-speaker models, e.g.
        models trained using the VCTK dataset. Not used for single-speaker
        models, e.g., models trained using the LJ speech dataset.
        """,
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
        "text",
        type=str,
        help="The input text to generate audio for",
    )

    return parser.parse_args()


# buffer saves audio samples to be played
buffer = queue.Queue()

# started is set to True once generated_audio_callback is called.
started = False

# stopped is set to True once all the text has been processed
stopped = False

# killed is set to True once ctrl + C is pressed
killed = False

# Note: When started is True, and stopped is True, and buffer is empty,
# we will exit the program since all audio samples have been played.

sample_rate = None

event = threading.Event()

first_message_time = None


def generated_audio_callback(samples: np.ndarray):
    """This function is called whenever max_num_sentences sentences
    have been processed.

    Note that it is passed to C++ and is invoked in C++.

    Args:
      samples:
        A 1-D np.float32 array containing audio samples
    """
    global first_message_time
    if first_message_time is None:
        first_message_time = time.time()

    buffer.put(samples)
    global started

    if started is False:
        logging.info("Start playing ...")
    started = True


# see https://python-sounddevice.readthedocs.io/en/0.4.6/api/streams.html#sounddevice.OutputStream
def play_audio_callback(
    outdata: np.ndarray, frames: int, time, status: sd.CallbackFlags
):
    if killed or (started and buffer.empty() and stopped):
        event.set()

    # outdata is of shape (frames, num_channels)
    if buffer.empty():
        outdata.fill(0)
        return

    n = 0
    while n < frames and not buffer.empty():
        remaining = frames - n
        k = buffer.queue[0].shape[0]

        if remaining <= k:
            outdata[n:, 0] = buffer.queue[0][:remaining]
            buffer.queue[0] = buffer.queue[0][remaining:]
            n = frames
            if buffer.queue[0].shape[0] == 0:
                buffer.get()

            break

        outdata[n : n + k, 0] = buffer.get()
        n += k

    if n < frames:
        outdata[n:, 0] = 0


# Please see
# https://python-sounddevice.readthedocs.io/en/0.4.6/usage.html#device-selection
# for how to select a device
def play_audio():
    if False:
        # This if branch can be safely removed. It is here to show you how to
        # change the default output device in case you need that.
        devices = sd.query_devices()
        print(devices)

        # sd.default.device[1] is the output device, if you want to
        # select a different device, say, 3, as the output device, please
        # use self.default.device[1] = 3

        default_output_device_idx = sd.default.device[1]
        print(
            f'Use default output device: {devices[default_output_device_idx]["name"]}'
        )

    with sd.OutputStream(
        channels=1,
        callback=play_audio_callback,
        dtype="float32",
        samplerate=sample_rate,
        blocksize=1024,
    ):
        event.wait()

    logging.info("Exiting ...")


def main():
    args = get_args()
    print(args)

    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=args.vits_model,
                lexicon=args.vits_lexicon,
                data_dir=args.vits_data_dir,
                tokens=args.vits_tokens,
            ),
            provider=args.provider,
            debug=args.debug,
            num_threads=args.num_threads,
        ),
        rule_fsts=args.tts_rule_fsts,
        max_num_sentences=1,
    )

    if not tts_config.validate():
        raise ValueError("Please check your config")

    logging.info("Loading model ...")
    tts = sherpa_onnx.OfflineTts(tts_config)
    logging.info("Loading model done.")

    global sample_rate
    sample_rate = tts.sample_rate

    play_back_thread = threading.Thread(target=play_audio)
    play_back_thread.start()

    logging.info("Start generating ...")
    start_time = time.time()
    audio = tts.generate(
        args.text,
        sid=args.sid,
        speed=args.speed,
        callback=generated_audio_callback,
    )
    end_time = time.time()
    logging.info("Finished generating!")
    global stopped
    stopped = True

    if len(audio.samples) == 0:
        print("Error in generating audios. Please read previous error messages.")
        global killed
        killed = True
        play_back_thread.join()
        return

    elapsed_seconds = end_time - start_time
    audio_duration = len(audio.samples) / audio.sample_rate
    real_time_factor = elapsed_seconds / audio_duration

    sf.write(
        args.output_filename,
        audio.samples,
        samplerate=audio.sample_rate,
        subtype="PCM_16",
    )
    logging.info(f"The text is '{args.text}'")
    logging.info(
        "Time in seconds to receive the first "
        f"message: {first_message_time-start_time:.3f}"
    )
    logging.info(f"Elapsed seconds: {elapsed_seconds:.3f}")
    logging.info(f"Audio duration in seconds: {audio_duration:.3f}")
    logging.info(
        f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}"
    )

    logging.info(f"***  Saved to {args.output_filename} ***")

    print("\n   >>>>>>>>> You can safely press ctrl + C to stop the play <<<<<<<<<<\n")

    play_back_thread.join()


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    try:
        main()
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting")
        killed = True
        sys.exit(0)
