# Example using the sherpa-onnx Python API and sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8 model
# Prints recognized text, per-token timestamps, and durations

import os
import sys
import sherpa_onnx
import soundfile as sf

wav_filename = "./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/test_wavs/en.wav"
encoder = "./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/encoder.int8.onnx"
decoder = "./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/decoder.int8.onnx"
joiner = "./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/joiner.int8.onnx"
tokens = "./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/tokens.txt"

if not os.path.exists(wav_filename):
    print(f"File not found: {wav_filename}")
    sys.exit(1)


recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder,
    decoder,
    joiner,
    tokens,
    num_threads=1,
    provider="cpu",
    debug=False,
    decoding_method="greedy_search",
    model_type="nemo_transducer"
)

audio, sample_rate = sf.read(wav_filename, dtype="float32", always_2d=True)
audio = audio[:, 0]  # use first channel if multi-channel
stream = recognizer.create_stream()
stream.accept_waveform(sample_rate, audio)
recognizer.decode_stream(stream)
result = stream.result

print(f"Recognized text: {result.text}")

if hasattr(result, "tokens") and hasattr(result, "timestamps") and hasattr(result, "durations"):
    print("Token\tTimestamp\tDuration")
    for token, ts, dur in zip(result.tokens, result.timestamps, result.durations):
        print(f"{token}\t{ts:.2f}\t{dur:.2f}")
else:
    print("Timestamps or durations not available.")
