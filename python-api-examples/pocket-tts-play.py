#!/usr/bin/env python3
#
# Copyright (c)  2026  Xiaomi Corporation

"""
This file demonstrates how to use sherpa-onnx Python API
for voice cloning using PocketTTS.

Different from ./pocket-tts.py, this file plays back the generated audio
while the model is still generating.

Usage:

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
tar xvf sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
rm sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2

python3 ./pocket-tts-play.py

You can find more models at
https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models

Please see
https://k2-fsa.github.io/sherpa/onnx/tts/pocket.html
for details.
"""

import logging
import queue
import sys
import threading
import time
from pathlib import Path

import librosa
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


def create_tts():
    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            pocket=sherpa_onnx.OfflineTtsPocketModelConfig(
                lm_flow="./sherpa-onnx-pocket-tts-int8-2026-01-26/lm_flow.int8.onnx",
                lm_main="./sherpa-onnx-pocket-tts-int8-2026-01-26/lm_main.int8.onnx",
                encoder="./sherpa-onnx-pocket-tts-int8-2026-01-26/encoder.onnx",
                decoder="./sherpa-onnx-pocket-tts-int8-2026-01-26/decoder.int8.onnx",
                text_conditioner="./sherpa-onnx-pocket-tts-int8-2026-01-26/text_conditioner.onnx",
                vocab_json="./sherpa-onnx-pocket-tts-int8-2026-01-26/vocab.json",
                token_scores_json="./sherpa-onnx-pocket-tts-int8-2026-01-26/token_scores.json",
            ),
            debug=True,  # set it to True to see verbose logs
            num_threads=2,
            provider="cpu",
        )
    )
    if not tts_config.validate():
        raise ValueError(
            "Please read the previous error messages and re-check your config"
        )

    return sherpa_onnx.OfflineTts(tts_config)


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


def generated_audio_callback(samples: np.ndarray, progress: float):
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

    # 1 means to keep generating
    # 0 means to stop generating
    if killed:
        return 0

    return 1


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
    reference_audio_file = "./sherpa-onnx-pocket-tts-int8-2026-01-26/test_wavs/bria.wav"
    if not Path(reference_audio_file).is_file():
        raise ValueError(f"Reference audio {reference_audio_file} does not exist")

    logging.info("Loading model ...")
    tts = create_tts()
    logging.info("Loading model done.")

    reference_audio, reference_sample_rate = librosa.load(
        reference_audio_file, sr=tts.sample_rate
    )

    text = """
    I am happy to join with you today in what will go down in history as the greatest demonstration for freedom in the history of our nation.
    Five score years ago, a great American, in whose symbolic shadow we stand today, signed the Emancipation Proclamation. This momentous decree came as a great beacon light of hope to millions of Negro slaves who had been seared in the flames of withering injustice. It came as a joyous daybreak to end the long night of their captivity.
    But one hundred years later, the Negro still is not free. One hundred years later, the life of the Negro is still sadly crippled by the manacles of segregation and the chains of discrimination. One hundred years later, the Negro lives on a lonely island of poverty in the midst of a vast ocean of material prosperity. One hundred years later, the Negro is still languished in the corners of American society and finds himself an exile in his own land. And so we've come here today to dramatize a shameful condition.
    In a sense we've come to our nation's capital to cash a check. When the architects of our republic wrote the magnificent words of the Constitution and the Declaration of Independence, they were signing a promissory note to which every American was to fall heir. This note was a promise that all men, yes, black men as well as white men, would be guaranteed the "unalienable Rights" of "Life, Liberty and the pursuit of Happiness." It is obvious today that America has defaulted on this promissory note, insofar as her citizens of color are concerned. Instead of honoring this sacred obligation, America has given the Negro people a bad check, a check which has come back marked insufficient funds.
    """

    global sample_rate
    sample_rate = tts.sample_rate

    gen_config = sherpa_onnx.GenerationConfig()
    gen_config.reference_audio = reference_audio
    gen_config.reference_sample_rate = reference_sample_rate
    gen_config.num_steps = 5

    play_back_thread = threading.Thread(target=play_audio)
    play_back_thread.start()

    logging.info("Start generating ...")
    start_time = time.time()
    audio = tts.generate(
        text,
        gen_config,
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

    output_filename = "./generated.wav"
    sf.write(
        output_filename,
        audio.samples,
        samplerate=audio.sample_rate,
        subtype="PCM_16",
    )
    logging.info(f"The text is '{text}'")
    logging.info(
        "Time in seconds to receive the first "
        f"message: {first_message_time-start_time:.3f}"
    )
    logging.info(f"Elapsed seconds: {elapsed_seconds:.3f}")
    logging.info(f"Audio duration in seconds: {audio_duration:.3f}")
    logging.info(
        f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}"
    )

    logging.info(f"***  Saved to {output_filename} ***")

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
