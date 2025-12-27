#!/usr/bin/env python3
# Copyright      2025  Xiaomi Corp.        (authors: Fangjun Kuang)

from torch import nn

import adaptor
import torch_model


class Nano(nn.Module):
    def __init__(self, vocab_size: int = 60515):
        super().__init__()
        self.audio_encoder = torch_model.SenseVoiceEncoderSmall()
        self.ctc_decoder = adaptor.Transformer()
        # blank is 60514, i.e., the last token id
        self.ctc = torch_model.CTC(
            odim=vocab_size,
            encoder_output_size=self.audio_encoder.output_size,
        )

    def forward(self, x):
        """
        Args:
          x: (N, T, C)
        Returns:
          - logits: (N, T, vocab_size)
        """
        encoder_out = self.audio_encoder(x)
        encoder_out = self.ctc_decoder(encoder_out)
        logits = self.ctc.ctc_lo(encoder_out)
        return logits
