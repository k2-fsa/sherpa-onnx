#!/usr/bin/env python3
# Copyright    2024  Xiaomi Corp.        (authors: Fangjun Kuang)

"""
Please refer to
https://github.com/k2-fsa/sherpa-onnx/blob/master/.github/workflows/speaker-diarization.yaml
for usages.
"""

import argparse
from datetime import timedelta
from pathlib import Path
from typing import List

import librosa
import numpy as np
import onnxruntime as ort
import sherpa_onnx
import soundfile as sf
from numpy.lib.stride_tricks import as_strided


class Segment:
    def __init__(
        self,
        start,
        end,
        speaker,
    ):
        assert start < end
        self.start = start
        self.end = end
        self.speaker = speaker

    def merge(self, other, gap=0.5):
        assert self.speaker == other.speaker, (self.speaker, other.speaker)
        if self.end < other.start and self.end + gap >= other.start:
            return Segment(start=self.start, end=other.end, speaker=self.speaker)
        elif other.end < self.start and other.end + gap >= self.start:
            return Segment(start=other.start, end=self.end, speaker=self.speaker)
        else:
            return None

    @property
    def duration(self):
        return self.end - self.start

    def __str__(self):
        s = f"{timedelta(seconds=self.start)}"[:-3]
        s += " --> "
        s += f"{timedelta(seconds=self.end)}"[:-3]
        s += f" speaker_{self.speaker:02d}"
        return s


def merge_segment_list(in_out: List[Segment], min_duration_off: float):
    changed = True
    while changed:
        changed = False
        for i in range(len(in_out)):
            if i + 1 >= len(in_out):
                continue

            new_segment = in_out[i].merge(in_out[i + 1], gap=min_duration_off)
            if new_segment is None:
                continue
            del in_out[i + 1]
            in_out[i] = new_segment
            changed = True
            break


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seg-model",
        type=str,
        required=True,
        help="Path to model.onnx for segmentation",
    )
    parser.add_argument(
        "--speaker-embedding-model",
        type=str,
        required=True,
        help="Path to model.onnx for speaker embedding extractor",
    )
    parser.add_argument("--wav", type=str, required=True, help="Path to test.wav")

    return parser.parse_args()


class OnnxSegmentationModel:
    def __init__(self, filename):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.session_opts = session_opts

        self.model = ort.InferenceSession(
            filename,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        meta = self.model.get_modelmeta().custom_metadata_map
        print(meta)

        self.window_size = int(meta["window_size"])
        self.sample_rate = int(meta["sample_rate"])
        self.window_shift = int(0.1 * self.window_size)
        self.receptive_field_size = int(meta["receptive_field_size"])
        self.receptive_field_shift = int(meta["receptive_field_shift"])
        self.num_speakers = int(meta["num_speakers"])
        self.powerset_max_classes = int(meta["powerset_max_classes"])
        self.num_classes = int(meta["num_classes"])

    def __call__(self, x):
        """
        Args:
          x: (N, num_samples)
        Returns:
          A tensor of shape (N, num_frames, num_classes)
        """
        x = np.expand_dims(x, axis=1)

        (y,) = self.model.run(
            [self.model.get_outputs()[0].name], {self.model.get_inputs()[0].name: x}
        )

        return y


def load_wav(filename, expected_sample_rate) -> np.ndarray:
    audio, sample_rate = sf.read(filename, dtype="float32", always_2d=True)
    audio = audio[:, 0]  # only use the first channel
    if sample_rate != expected_sample_rate:
        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=expected_sample_rate,
        )
    return audio


def get_powerset_mapping(num_classes, num_speakers, powerset_max_classes):
    mapping = np.zeros((num_classes, num_speakers))

    k = 1
    for i in range(1, powerset_max_classes + 1):
        if i == 1:
            for j in range(0, num_speakers):
                mapping[k, j] = 1
                k += 1
        elif i == 2:
            for j in range(0, num_speakers):
                for m in range(j + 1, num_speakers):
                    mapping[k, j] = 1
                    mapping[k, m] = 1
                    k += 1
        elif i == 3:
            raise RuntimeError("Unsupported")

    return mapping


def to_multi_label(y, mapping):
    """
    Args:
      y: (num_chunks, num_frames, num_classes)
    Returns:
      A tensor of shape (num_chunks, num_frames, num_speakers)
    """
    y = np.argmax(y, axis=-1)
    labels = mapping[y.reshape(-1)].reshape(y.shape[0], y.shape[1], -1)
    return labels


# speaker count per frame
def speaker_count(labels, seg_m):
    """
    Args:
      labels: (num_chunks, num_frames, num_speakers)
      seg_m: Segmentation model
    Returns:
      A integer array of shape (num_total_frames,)
    """
    labels = labels.sum(axis=-1)
    # Now labels: (num_chunks, num_frames)

    num_frames = (
        int(
            (seg_m.window_size + (labels.shape[0] - 1) * seg_m.window_shift)
            / seg_m.receptive_field_shift
        )
        + 1
    )
    ans = np.zeros((num_frames,))
    count = np.zeros((num_frames,))

    for i in range(labels.shape[0]):
        this_chunk = labels[i]
        start = int(i * seg_m.window_shift / seg_m.receptive_field_shift + 0.5)
        end = start + this_chunk.shape[0]
        ans[start:end] += this_chunk
        count[start:end] += 1

    ans /= np.maximum(count, 1e-12)

    return (ans + 0.5).astype(np.int8)


def load_speaker_embedding_model(filename):
    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=filename,
        num_threads=1,
        debug=0,
    )
    if not config.validate():
        raise ValueError(f"Invalid config. {config}")
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
    return extractor


def get_embeddings(embedding_filename, audio, labels, seg_m, exclude_overlap):
    """
    Args:
      embedding_filename: Path to the speaker embedding extractor model
      audio: (num_samples,)
      labels: (num_chunks, num_frames, num_speakers)
      seg_m: segmentation model
    Returns:
      Return (num_chunks, num_speakers, embedding_dim)
    """
    if exclude_overlap:
        labels = labels * (labels.sum(axis=-1, keepdims=True) < 2)

    extractor = load_speaker_embedding_model(embedding_filename)
    buffer = np.empty(seg_m.window_size)
    num_chunks, num_frames, num_speakers = labels.shape

    ans_chunk_speaker_pair = []
    ans_embeddings = []

    for i in range(num_chunks):
        labels_T = labels[i].T
        # t: (num_speakers, num_frames)

        sample_offset = i * seg_m.window_shift

        for j in range(num_speakers):
            frames = labels_T[j]
            if frames.sum() < 10:
                # skip segment less than 20 frames, i.e., about 0.2 seconds
                continue

            start = None
            start_samples = 0
            idx = 0
            for k in range(num_frames):
                if frames[k] != 0:
                    if start is None:
                        start = k
                elif start is not None:
                    start_samples = (
                        int(start / num_frames * seg_m.window_size) + sample_offset
                    )
                    end_samples = (
                        int(k / num_frames * seg_m.window_size) + sample_offset
                    )
                    num_samples = end_samples - start_samples
                    buffer[idx : idx + num_samples] = audio[start_samples:end_samples]
                    idx += num_samples

                    start = None
            if start is not None:
                start_samples = (
                    int(start / num_frames * seg_m.window_size) + sample_offset
                )
                end_samples = int(k / num_frames * seg_m.window_size) + sample_offset
                num_samples = end_samples - start_samples
                buffer[idx : idx + num_samples] = audio[start_samples:end_samples]
                idx += num_samples

            stream = extractor.create_stream()
            stream.accept_waveform(sample_rate=seg_m.sample_rate, waveform=buffer[:idx])
            stream.input_finished()

            assert extractor.is_ready(stream)
            embedding = extractor.compute(stream)
            embedding = np.array(embedding)

            ans_chunk_speaker_pair.append([i, j])
            ans_embeddings.append(embedding)

    assert len(ans_chunk_speaker_pair) == len(ans_embeddings), (
        len(ans_chunk_speaker_pair),
        len(ans_embeddings),
    )
    return ans_chunk_speaker_pair, np.array(ans_embeddings)


def main():
    args = get_args()
    assert Path(args.seg_model).is_file(), args.seg_model
    assert Path(args.wav).is_file(), args.wav

    seg_m = OnnxSegmentationModel(args.seg_model)
    audio = load_wav(args.wav, seg_m.sample_rate)
    # audio: (num_samples,)

    num = (audio.shape[0] - seg_m.window_size) // seg_m.window_shift + 1

    samples = as_strided(
        audio,
        shape=(num, seg_m.window_size),
        strides=(seg_m.window_shift * audio.strides[0], audio.strides[0]),
    )

    # or use torch.Tensor.unfold
    #  samples = torch.from_numpy(audio).unfold(0, seg_m.window_size, seg_m.window_shift).numpy()

    if (
        audio.shape[0] < seg_m.window_size
        or (audio.shape[0] - seg_m.window_size) % seg_m.window_shift > 0
    ):
        has_last_chunk = True
    else:
        has_last_chunk = False

    num_chunks = samples.shape[0]
    batch_size = 32
    output = []
    for i in range(0, num_chunks, batch_size):
        start = i
        end = i + batch_size
        # it's perfectly ok to use end > num_chunks
        y = seg_m(samples[start:end])
        output.append(y)

    if has_last_chunk:
        last_chunk = audio[num_chunks * seg_m.window_shift :]  # noqa
        pad_size = seg_m.window_size - last_chunk.shape[0]
        last_chunk = np.pad(last_chunk, (0, pad_size))
        last_chunk = np.expand_dims(last_chunk, axis=0)
        y = seg_m(last_chunk)
        output.append(y)

    y = np.vstack(output)
    # y: (num_chunks, num_frames, num_classes)

    mapping = get_powerset_mapping(
        num_classes=seg_m.num_classes,
        num_speakers=seg_m.num_speakers,
        powerset_max_classes=seg_m.powerset_max_classes,
    )
    labels = to_multi_label(y, mapping=mapping)
    # labels: (num_chunks, num_frames, num_speakers)

    inactive = (labels.sum(axis=1) == 0).astype(np.int8)
    # inactive: (num_chunks, num_speakers)

    speakers_per_frame = speaker_count(labels=labels, seg_m=seg_m)
    # speakers_per_frame: (num_frames, speakers_per_frame)

    if speakers_per_frame.max() == 0:
        print("No speakers found in the audio file!")
        return

    # if users specify only 1 speaker for clustering, then return the
    # result directly

    # Now, get embeddings
    chunk_speaker_pair, embeddings = get_embeddings(
        args.speaker_embedding_model,
        audio=audio,
        labels=labels,
        seg_m=seg_m,
        #  exclude_overlap=True,
        exclude_overlap=False,
    )
    # chunk_speaker_pair: a list of (chunk_idx, speaker_idx)
    # embeddings: (batch_size, embedding_dim)

    # Please change num_clusters or threshold by yourself.
    clustering_config = sherpa_onnx.FastClusteringConfig(num_clusters=2)
    #  clustering_config = sherpa_onnx.FastClusteringConfig(threshold=0.8)
    clustering = sherpa_onnx.FastClustering(clustering_config)
    cluster_labels = clustering(embeddings)

    chunk_speaker_to_cluster = dict()
    for (chunk_idx, speaker_idx), cluster_idx in zip(
        chunk_speaker_pair, cluster_labels
    ):
        if inactive[chunk_idx, speaker_idx] == 1:
            print("skip ", chunk_idx, speaker_idx)
            continue
        chunk_speaker_to_cluster[(chunk_idx, speaker_idx)] = cluster_idx

    num_speakers = max(cluster_labels) + 1
    relabels = np.zeros((labels.shape[0], labels.shape[1], num_speakers))
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            for k in range(labels.shape[2]):
                if (i, k) not in chunk_speaker_to_cluster:
                    continue
                t = chunk_speaker_to_cluster[(i, k)]

                if labels[i, j, k] == 1:
                    relabels[i, j, t] = 1

    num_frames = (
        int(
            (seg_m.window_size + (relabels.shape[0] - 1) * seg_m.window_shift)
            / seg_m.receptive_field_shift
        )
        + 1
    )

    count = np.zeros((num_frames, relabels.shape[-1]))
    for i in range(relabels.shape[0]):
        this_chunk = relabels[i]
        start = int(i * seg_m.window_shift / seg_m.receptive_field_shift + 0.5)
        end = start + this_chunk.shape[0]
        count[start:end] += this_chunk

    if has_last_chunk:
        stop_frame = int(audio.shape[0] / seg_m.receptive_field_shift)
        count = count[:stop_frame]

    sorted_count = np.argsort(-count, axis=-1)
    final = np.zeros((count.shape[0], count.shape[1]))

    for i, (c, sc) in enumerate(zip(speakers_per_frame, sorted_count)):
        for k in range(c):
            final[i, sc[k]] = 1

    min_duration_off = 0.5
    min_duration_on = 0.3
    onset = 0.5
    offset = 0.5
    # final: (num_frames, num_speakers)

    final = final.T
    for kk in range(final.shape[0]):
        segment_list = []
        frames = final[kk]

        is_active = frames[0] > onset

        start = None
        if is_active:
            start = 0
        scale = seg_m.receptive_field_shift / seg_m.sample_rate
        scale_offset = seg_m.receptive_field_size / seg_m.sample_rate * 0.5
        for i in range(1, len(frames)):
            if is_active:
                if frames[i] < offset:
                    segment = Segment(
                        start=start * scale + scale_offset,
                        end=i * scale + scale_offset,
                        speaker=kk,
                    )
                    segment_list.append(segment)
                    is_active = False
            else:
                if frames[i] > onset:
                    start = i
                    is_active = True

        if is_active:
            segment = Segment(
                start=start * scale + scale_offset,
                end=(len(frames) - 1) * scale + scale_offset,
                speaker=kk,
            )
            segment_list.append(segment)

        if len(segment_list) > 1:
            merge_segment_list(segment_list, min_duration_off=min_duration_off)
            for s in segment_list:
                if s.duration < min_duration_on:
                    continue
                print(s)


if __name__ == "__main__":
    main()
