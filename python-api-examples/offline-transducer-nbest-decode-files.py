#!/usr/bin/env python3

"""
This file shows how to get the n-best hypotheses from a non-streaming
transducer model decoded with modified_beam_search.

Set num_return_paths to a value greater than 1 and read result.hypotheses.
It is ordered from best to worst, and hypotheses[0] matches the top-level
result.text / result.tokens / result.timestamps.

Please download model files from
https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models

For instance,

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-small-en-2023-06-26.tar.bz2
tar xvf sherpa-onnx-zipformer-small-en-2023-06-26.tar.bz2
rm sherpa-onnx-zipformer-small-en-2023-06-26.tar.bz2
"""

from pathlib import Path

import sherpa_onnx
import soundfile as sf


def create_recognizer():
    d = "./sherpa-onnx-zipformer-small-en-2023-06-26"

    encoder = f"{d}/encoder-epoch-99-avg-1.onnx"
    decoder = f"{d}/decoder-epoch-99-avg-1.onnx"
    joiner = f"{d}/joiner-epoch-99-avg-1.onnx"
    tokens = f"{d}/tokens.txt"
    test_wav = f"{d}/test_wavs/0.wav"

    for f in [encoder, decoder, joiner, tokens, test_wav]:
        if not Path(f).is_file():
            raise ValueError(f"{f} does not exist. Please read the comments.")

    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        tokens=tokens,
        decoding_method="modified_beam_search",
        # the search width
        max_active_paths=8,
        # how many of those paths to return; clamped to max_active_paths
        num_return_paths=5,
    )

    return recognizer, test_wav


def main():
    recognizer, wave_filename = create_recognizer()

    audio, sample_rate = sf.read(wave_filename, dtype="float32", always_2d=True)
    audio = audio[:, 0]  # only use the first channel

    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, audio)
    recognizer.decode_stream(stream)

    result = stream.result

    print("1-best:")
    print(f"  {result.text}")

    print(f"\nn-best ({len(result.hypotheses)} hypotheses):")
    for i, hyp in enumerate(result.hypotheses):
        print(f"  [{i}] score={hyp.score:.4f}")
        print(f"      {hyp.text}")


if __name__ == "__main__":
    main()
