#!/usr/bin/env python3

from torch import nn

from adaptor import Transformer
from torch_model import CTC, SenseVoiceEncoderSmall


class Nano(nn.Module):
    def __init__(self, vocab_size: int = 60515):
        super().__init__()
        self.audio_encoder = SenseVoiceEncoderSmall()
        self.ctc_decoder = Transformer()
        # blank is 60514, i.e., the last token id
        self.ctc = CTC(
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
