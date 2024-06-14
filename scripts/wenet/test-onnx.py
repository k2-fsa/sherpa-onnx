#!/usr/bin/env python3
# Copyright      2023  Xiaomi Corp.        (authors: Fangjun Kuang)

import kaldi_native_fbank as knf
import onnxruntime as ort
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence


class OnnxModel:
    def __init__(
        self,
        filename: str,
    ):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 4

        self.session_opts = session_opts

        self.model = ort.InferenceSession(
            filename,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

    def __call__(self, x: torch.Tensor, x_lens: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C)
          x_lens:
            A 1-D tensor of shape (N,). Its dtype is torch.int64
        Returns:
          Return a 3-D tensor of shape (N, T, C) containing log_probs.
        """
        log_probs, log_probs_lens = self.model.run(
            [self.model.get_outputs()[0].name, self.model.get_outputs()[1].name],
            {
                self.model.get_inputs()[0].name: x.numpy(),
                self.model.get_inputs()[1].name: x_lens.numpy(),
            },
        )
        return torch.from_numpy(log_probs), torch.from_numpy(log_probs_lens)


def get_features(test_wav_filename):
    wave, sample_rate = torchaudio.load(test_wav_filename)
    audio = wave[0].contiguous()  # only use the first channel
    if sample_rate != 16000:
        audio = torchaudio.functional.resample(
            audio, orig_freq=sample_rate, new_freq=16000
        )
    audio *= 32768

    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.mel_opts.num_bins = 80
    opts.frame_opts.snip_edges = False
    opts.mel_opts.debug_mel = False

    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(16000, audio.numpy())
    frames = []
    for i in range(fbank.num_frames_ready):
        frames.append(torch.from_numpy(fbank.get_frame(i)))
    frames = torch.stack(frames)
    return frames


def main():
    model_filename = "./model.onnx"
    model = OnnxModel(model_filename)

    filename = "./0.wav"
    x = get_features(filename)
    x = x.unsqueeze(0)

    # Note: It supports only batch size == 1
    x_lens = torch.tensor([x.shape[1]], dtype=torch.int64)

    print(x.shape, x_lens)

    log_probs, log_probs_lens = model(x, x_lens)
    log_probs = log_probs[0]
    print(log_probs.shape)

    indexes = log_probs.argmax(dim=1)
    print(indexes)
    indexes = torch.unique_consecutive(indexes)
    indexes = indexes[indexes != 0].tolist()

    id2word = dict()
    with open("./units.txt", encoding="utf-8") as f:
        for line in f:
            word, idx = line.strip().split()
            id2word[int(idx)] = word
    text = "".join([id2word[i] for i in indexes])
    print(text)


if __name__ == "__main__":
    main()
