"""
This script shows how to use Python APIs for speaker identification with
a microphone.

Usage:

(1) Prepare a text file containing speaker related files.

Each line in the text file contains two columns. The first column is the
speaker name, while the second column contains the wave file of the speaker.

If the text file contains multiple wave files for the same speaker, then the
embeddings of these files are averaged.

An example text file is given below:

    foo /path/to/a.wav
    bar /path/to/b.wav
    foo /path/to/c.wav
    foobar /path/to/d.wav

Each wave file should contain only a single channel; the sample format
should be int16_t; the sample rate can be arbitrary.

(2) Download a model for computing speaker embeddings

Please visit
https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
to download a model. An example is given below:

    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_zh_cnceleb_resnet34.onnx

Note that `zh` means Chinese, while `en` means English.

(3) Run this script

Assume the filename of the text file is speaker.txt.

python3 ./python-api-examples/speaker-identification.py \
  --speaker-file ./speaker.txt \
  --model ./wespeaker_zh_cnceleb_resnet34.onnx
"""

import functools
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Union

import numpy as np
import sherpa_onnx
import soundfile as sf
from numpy.typing import NDArray


def load_speaker_embedding_model(
    model: Union[str, Path],
    num_threads: int,
    debug: bool,
    provider: Literal["cpu", "cuda", "coreml"],
) -> sherpa_onnx.SpeakerEmbeddingExtractor:
    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(model),
        num_threads=num_threads,
        debug=debug,
        provider=provider,
    )
    if not config.validate():
        raise ValueError(f"Invalid config. {config}")
    return sherpa_onnx.SpeakerEmbeddingExtractor(config)


def register_speaker(
    speaker_file: Union[str, Path],
    extractor: sherpa_onnx.SpeakerEmbeddingExtractor,
    manager: sherpa_onnx.SpeakerEmbeddingManager,
) -> None:
    speaker_table = load_speaker_file(speaker_file)

    for name, filename_list in speaker_table.items():
        embedding = compute_avg_speaker_embedding(
            filenames=filename_list, extractor=extractor
        )
        status = manager.add(name, embedding)
        if not status:
            raise RuntimeError(f"Failed to register speaker {name}")


def load_speaker_file(speaker_file: Union[str, Path]) -> Dict[str, List[Path]]:
    """Load and parse the speaker audio file list.

    Example:
        speaker.txt:
            Alice /home/user/voices/alice_1.wav
            Bob /home/user/voices/bob_1.wav
            Alice /home/user/voices/alice_2.wav

        Returns:
            {
                'Alice': [
                    Path('/home/user/voices/alice_1.wav'),
                    Path('/home/user/voices/alice_2.wav')
                ],
                'Bob': [
                    Path('/home/user/voices/bob_1.wav')
                ]
            }
    """
    speaker_file = Path(speaker_file)

    if not speaker_file.is_file():
        raise ValueError(f"--speaker-file {speaker_file} does not exist")

    table = defaultdict(list)
    with speaker_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f"Invalid line: {line}. Fields: {fields}")

            speaker_name, audio_path = fields
            table[speaker_name].append(Path(audio_path))
    return table


def load_audio(filename: Union[str, Path]) -> Tuple[NDArray[np.float32], int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    samples = np.ascontiguousarray(data, dtype=np.float32)
    return samples, sample_rate


def compute_speaker_embedding(
    filename: Union[str, Path],
    extractor: sherpa_onnx.SpeakerEmbeddingExtractor,
) -> NDArray[np.float32]:
    samples, sample_rate = load_audio(filename)
    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
    stream.input_finished()

    assert extractor.is_ready(stream)
    embedding: List[float] = extractor.compute(stream)
    return np.array(embedding, dtype=np.float32)


def compute_avg_speaker_embedding(
    filenames: Union[List[str], List[Path]],
    extractor: sherpa_onnx.SpeakerEmbeddingExtractor,
) -> NDArray[np.float32]:
    assert len(filenames) > 0, "filenames is empty"

    compute_emb = functools.partial(compute_speaker_embedding, extractor=extractor)
    embeddings_sum = np.zeros(extractor.dim, dtype=np.float32)
    for filename in filenames:
        print(f"processing {filename}")
        embeddings_sum += compute_emb(filename)

    return embeddings_sum / len(filenames)


# %%
# The following code is required for command line interface.
# If you only need the packaged functions, you can use only the code above
import argparse
import queue
import threading
from typing import Optional

try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice first. You can use\n\t")
    print("pip install sounddevice")
    print("\nto install it")
    import sys

    sys.exit(1)


class Args(argparse.Namespace):
    speaker_file: Path
    model: Path
    threshold: float
    num_threads: int
    debug: bool
    provider: Literal["cpu", "cuda", "coreml"]


def get_args() -> Args:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--speaker-file",
        type=Path,
        required=True,
        help="""Path to the speaker file. Read the help doc at the beginning of this
        file for the format.""",
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the model file.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads for neural network computation",
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
        choices=["cpu", "cuda", "coreml"],
        help="Valid values: cpu, cuda, coreml",
    )

    return parser.parse_args(namespace=Args())


class AudioRecorder:
    @staticmethod
    def list_audio_devices() -> None:
        """List all available audio input devices."""
        print("=" * 50)
        print(sd.query_devices())
        print("=" * 50)

    def __init__(
        self, device_index: Union[int, None] = None, sample_rate: int = 16000
    ) -> None:
        self.device_index = device_index  # None represents the default device
        self.sample_rate: int = sample_rate
        self._buffer: queue.Queue[NDArray[np.float32]] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self.print_microphone_device_info()

    def print_microphone_device_info(self) -> None:
        try:
            device_info = sd.query_devices(device=self.device_index, kind="input")
        except sd.PortAudioError as e:
            raise RuntimeError("No microphone devices found") from e

        print("=" * 50)
        print("Microphone device information:\n")
        print(f"Device ID: {device_info['index']}")
        print(f"Name: {device_info['name']}")
        print(f"Default Channels: {device_info['max_input_channels']}")
        print(f"Default SampleRate: {device_info['default_samplerate']}")
        print("=" * 50)

    def read_mic(self) -> None:
        print("Please speak!")
        samples_per_read = int(0.1 * self.sample_rate)
        with sd.InputStream(
            device=self.device_index,
            channels=1,
            samplerate=self.sample_rate,
            dtype="float32",
        ) as s:
            while not self._stop_event.is_set():
                samples, _ = s.read(samples_per_read)  # a blocking read
                self._buffer.put(samples)

    def start_recording(self) -> None:
        self._stop_event.clear()
        self._buffer.queue.clear()
        self._thread = threading.Thread(target=self.read_mic, daemon=True)
        self._thread.start()

    def stop_recording(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join()

    def infer_speaker(
        self,
        extractor: sherpa_onnx.SpeakerEmbeddingExtractor,
        manager: sherpa_onnx.SpeakerEmbeddingManager,
        threshold: float,
    ) -> str:
        stream = extractor.create_stream()
        while not self._buffer.empty():
            samples = self._buffer.get()
            stream.accept_waveform(sample_rate=self.sample_rate, waveform=samples)
        stream.input_finished()

        embedding = np.array(extractor.compute(stream), dtype=np.float32)
        name = manager.search(embedding, threshold=threshold)
        return name or "unknown"


def main() -> None:
    args = get_args()
    print(args)

    extractor = load_speaker_embedding_model(
        model=args.model,
        num_threads=args.num_threads,
        debug=args.debug,
        provider=args.provider,
    )
    manager = sherpa_onnx.SpeakerEmbeddingManager(extractor.dim)

    register_speaker(args.speaker_file, extractor, manager)

    recorder = AudioRecorder()
    while True:
        key = input("Press Enter to start recording")
        if key.lower() in {"q", "quit"}:
            break

        recorder.start_recording()
        input("Press Enter to stop recording")
        recorder.stop_recording()

        print("Compute embedding")
        name = recorder.infer_speaker(extractor, manager, args.threshold)
        print(f"Predicted name: {name}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting")
    except Exception as e:
        print(e)
        raise
