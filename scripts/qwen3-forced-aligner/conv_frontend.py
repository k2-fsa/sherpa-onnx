#!/usr/bin/env python3
#
# Conv frontend (Conv kept outside ONNX for INT8 encoder backend).

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pick(obj, names: List[str]):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


def _conv_out_len_3x_stride2(n: int) -> int:
    x = int(n)
    x = (x + 1) // 2
    x = (x + 1) // 2
    x = (x + 1) // 2
    return int(x)


def _proc_audio_tokens_len_int(n: int) -> int:
    leave = int(n) % 100
    feat = (leave - 1) // 2 + 1
    out = ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (int(n) // 100) * 13
    return int(out)


def _aftercnn(x: int) -> int:
    x = (x - 1) // 2 + 1
    x = (x - 1) // 2 + 1
    return (x - 1) // 2 + 1


def _feat_to_audio_tokens_len(feat_len, chunk_size: int = 100):
    import torch

    if isinstance(feat_len, torch.Tensor):
        cs = int(chunk_size)
        n = feat_len.to(torch.int64)
        full = n // cs
        rem = n % cs
        tn = _conv_out_len_3x_stride2(cs)
        out = full * tn + torch.tensor(
            [_aftercnn(int(r)) for r in rem], device=n.device
        )
        return torch.clamp(out, min=0).to(torch.int64)
    else:
        n = int(feat_len)
        cs = int(chunk_size)
        full = n // cs
        rem = n % cs
        tn = _conv_out_len_3x_stride2(cs)
        out = full * tn + _aftercnn(rem)
        return max(out, 0)


class ConvFrontend(nn.Module):
    def __init__(self, audio_tower: nn.Module, chunk_size: int = 100):
        super().__init__()
        self.audio_tower = audio_tower

        cfg = getattr(audio_tower, "config", None)
        self.n_window = (
            int(getattr(cfg, "n_window", 100)) if cfg is not None else 100
        )
        self.n_window_infer = (
            int(getattr(cfg, "n_window_infer", 400)) if cfg is not None else 400
        )
        self.conv_chunksize = (
            int(getattr(cfg, "conv_chunksize", 500)) if cfg is not None else 500
        )

        if int(chunk_size) <= 0:
            cands = [max(1, self.n_window * 2), max(1, self.n_window), 100]
            seen = set()
            cands = [
                c
                for c in cands
                if (c > 0 and (c not in seen) and not seen.add(c))
            ]
            chosen = cands[0]
            for cs in cands:
                if _conv_out_len_3x_stride2(cs) == _proc_audio_tokens_len_int(
                    cs
                ):
                    chosen = cs
                    break
            self.chunk_size = int(chosen)
        else:
            self.chunk_size = int(chunk_size)

        convs = []
        for cand in (
            ["conv2d1", "conv2d2", "conv2d3"],
            ["conv1", "conv2", "conv3"],
        ):
            tmp = []
            ok = True
            for nm in cand:
                m = _pick(self.audio_tower, [nm])
                if m is None:
                    ok = False
                    break
                tmp.append(m)
            if ok:
                convs = tmp
                break
        if not convs and hasattr(self.audio_tower, "convs"):
            convs = list(self.audio_tower.convs)
        if not convs:
            for name in dir(self.audio_tower):
                if name.startswith("conv") and not name.startswith("conv_out"):
                    m = getattr(self.audio_tower, name)
                    if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                        convs.append(m)
        if convs:
            convs.sort(
                key=lambda x: (
                    getattr(x, "weight", torch.zeros(1)).shape[0]
                    if hasattr(x, "weight")
                    else 0
                )
            )
        if not convs:
            raise RuntimeError(
                f"Cannot find conv frontend layers in {type(self.audio_tower)}"
            )
        self.convs = nn.ModuleList(convs)

        self.conv_out = _pick(
            self.audio_tower, ["conv_out", "proj", "linear_out"]
        )
        if self.conv_out is None:
            raise RuntimeError("Cannot find audio_tower.conv_out")

        self.tokens_per_chunk = _conv_out_len_3x_stride2(self.chunk_size)
        base = max(1, self.n_window * 2)
        ratio = max(1, self.n_window_infer // base)
        self.window_aftercnn = int(self.tokens_per_chunk * ratio)

    def forward(self, mel_bt_f: torch.Tensor) -> torch.Tensor:
        B, T, Fdim = mel_bt_f.shape
        cs = self.chunk_size

        pad_len = (cs - (T % cs)) % cs
        mel_bt_f = F.pad(mel_bt_f, (0, 0, 0, pad_len))

        Tpad = T + pad_len
        num_chunks = Tpad // cs

        x = mel_bt_f.view(B, num_chunks, cs, Fdim)
        x = (
            x.view(B * num_chunks, cs, Fdim).transpose(1, 2).unsqueeze(1)
        )  # (Bn,1,F,T)

        if self.conv_chunksize > 0 and (B * num_chunks) > self.conv_chunksize:
            outs = []
            for chunk in x.split(self.conv_chunksize, dim=0):
                y = F.gelu(self.convs[0](chunk))
                y = F.gelu(self.convs[1](y))
                y = F.gelu(self.convs[2](y))
                outs.append(y)
            x = torch.cat(outs, dim=0)
        else:
            x = F.gelu(self.convs[0](x))
            x = F.gelu(self.convs[1](x))
            x = F.gelu(self.convs[2](x))

        Bn, C, Freq, Tn = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(Bn, Tn, C * Freq)
        x = self.conv_out(x)
        x = x.view(B, num_chunks * Tn, -1)
        return x

    @property
    def output_dim(self) -> int:
        return int(self.conv_out.weight.shape[0])
