# This file is modified from
# https://github.com/modelscope/FunASR/blob/main/funasr/models/sense_voice/model.py

import torch
import torch.nn
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEncoder(nn.Module):
    def __init__(self, d_model=80, dropout_rate=0.1):
        super().__init__()

    def encode(
        self,
        positions: torch.Tensor = None,
        depth: int = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
          positions: (batch_size, )
        """
        batch_size = positions.size(0)
        positions = positions.type(dtype)
        device = positions.device
        log_timescale_increment = torch.log(
            torch.tensor([10000], dtype=dtype, device=device)
        ) / (depth / 2 - 1)
        inv_timescales = torch.exp(
            torch.arange(depth / 2, device=device).type(dtype)
            * (-log_timescale_increment)
        )
        inv_timescales = torch.reshape(inv_timescales, [batch_size, -1])
        scaled_time = torch.reshape(positions, [1, -1, 1]) * torch.reshape(
            inv_timescales, [1, 1, -1]
        )
        encoding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)
        return encoding.type(dtype)

    def forward(self, x):
        batch_size, timesteps, input_dim = x.size()
        positions = torch.arange(1, timesteps + 1, device=x.device)[None, :]
        position_encoding = self.encode(positions, input_dim, x.dtype).to(x.device)

        return x + position_encoding


class PositionwiseFeedForward(nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim, hidden_units, dropout_rate, activation=None):
        super().__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        if activation is None:
            activation = torch.nn.ReLU()
        self.activation = activation

    def forward(self, x):
        """Forward function."""
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class MultiHeadedAttentionSANM(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(
        self,
        n_head,
        in_feat,
        n_feat,
        dropout_rate,
        kernel_size,
        sanm_shfit=0,
        lora_list=None,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
    ):
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.linear_q_k_v = nn.Linear(in_feat, n_feat * 3)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

        self.fsmn_block = nn.Conv1d(
            n_feat, n_feat, kernel_size, stride=1, padding=0, groups=n_feat, bias=False
        )
        # padding
        left_padding = (kernel_size - 1) // 2
        if sanm_shfit > 0:
            left_padding = left_padding + sanm_shfit
        right_padding = kernel_size - 1 - left_padding
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)

    def forward_fsmn(self, inputs, mask, mask_shfit_chunk=None):
        b, t, d = inputs.size()
        if mask is not None:
            mask = torch.reshape(mask, (b, -1, 1))
            if mask_shfit_chunk is not None:
                mask = mask * mask_shfit_chunk
            inputs = inputs * mask

        x = inputs.transpose(1, 2)
        x = self.pad_fn(x)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        x += inputs
        x = self.dropout(x)
        if mask is not None:
            x = x * mask
        return x

    def forward_qkv(self, x):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        b, t, d = x.size()
        q_k_v = self.linear_q_k_v(x)
        q, k, v = torch.split(q_k_v, int(self.h * self.d_k), dim=-1)
        q_h = torch.reshape(q, (b, t, self.h, self.d_k)).transpose(
            1, 2
        )  # (batch, head, time1, d_k)
        k_h = torch.reshape(k, (b, t, self.h, self.d_k)).transpose(
            1, 2
        )  # (batch, head, time2, d_k)
        v_h = torch.reshape(v, (b, t, self.h, self.d_k)).transpose(
            1, 2
        )  # (batch, head, time2, d_k)

        return q_h, k_h, v_h, v

    def forward_attention(self, value, scores, mask, mask_att_chunk_encoder=None):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            if mask_att_chunk_encoder is not None:
                mask = mask * mask_att_chunk_encoder

            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)

            min_value = -float(
                "inf"
            )  # float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, x, mask, mask_shfit_chunk=None, mask_att_chunk_encoder=None):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q_h, k_h, v_h, v = self.forward_qkv(x)
        fsmn_memory = self.forward_fsmn(v, mask, mask_shfit_chunk)
        q_h = q_h * self.d_k ** (-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        att_outs = self.forward_attention(v_h, scores, mask, mask_att_chunk_encoder)
        return att_outs + fsmn_memory


class EncoderLayerSANM(nn.Module):
    def __init__(
        self,
        in_size,
        size,
        self_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        stochastic_depth_rate=0.0,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(in_size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.in_size = in_size
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate
        self.dropout_rate = dropout_rate

    def forward(
        self, x, mask, cache=None, mask_shfit_chunk=None, mask_att_chunk_encoder=None
    ):
        """Compute encoded features.

        Args:
            x_input (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """
        skip_layer = False
        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.
        stoch_layer_coeff = 1.0
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)

        if skip_layer:
            if cache is not None:
                x = torch.cat([cache, x], dim=1)
            return x, mask

        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if self.concat_after:
            x_concat = torch.cat(
                (
                    x,
                    self.self_attn(
                        x,
                        mask,
                        mask_shfit_chunk=mask_shfit_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    ),
                ),
                dim=-1,
            )
            if self.in_size == self.size:
                x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
            else:
                x = stoch_layer_coeff * self.concat_linear(x_concat)
        else:
            if self.in_size == self.size:
                x = residual + stoch_layer_coeff * self.dropout(
                    self.self_attn(
                        x,
                        mask,
                        mask_shfit_chunk=mask_shfit_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    )
                )
            else:
                x = stoch_layer_coeff * self.dropout(
                    self.self_attn(
                        x,
                        mask,
                        mask_shfit_chunk=mask_shfit_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    )
                )
                return x, mask
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + stoch_layer_coeff * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        return x, mask, cache, mask_shfit_chunk, mask_att_chunk_encoder


class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


def sequence_mask(lengths, maxlen=None, dtype=torch.float32, device=None):
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1).to(lengths.device)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix
    mask = mask.detach()

    return mask.type(dtype).to(device) if device is not None else mask.type(dtype)


class SenseVoiceEncoderSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = 80 * 7
        self.output_size = 512
        self.attention_heads = 4
        self.linear_units = 2048
        self.num_blocks = 50
        self.tp_blocks = 20
        self.input_layer = "pe"
        self.pos_enc_class = "SinusoidalPositionEncoder"
        self.normalize_before = True
        self.kernel_size = 11
        self.sanm_shfit = 0
        self.concat_after = False
        self.positionwise_layer_type = "linear"
        self.positionwise_conv_kernel_size = 1
        self.padding_idx = -1
        self.selfattention_layer_type = "sanm"
        self.dropout_rate = 0.1
        self.attention_dropout_rate = 0.1

        self._output_size = self.output_size

        self.embed = SinusoidalPositionEncoder()

        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            self.output_size,
            self.linear_units,
            self.dropout_rate,
        )

        encoder_selfattn_layer = MultiHeadedAttentionSANM
        encoder_selfattn_layer_args0 = (
            self.attention_heads,
            self.input_size,
            self.output_size,
            self.attention_dropout_rate,
            self.kernel_size,
            self.sanm_shfit,
        )
        encoder_selfattn_layer_args = (
            self.attention_heads,
            self.output_size,
            self.output_size,
            self.attention_dropout_rate,
            self.kernel_size,
            self.sanm_shfit,
        )

        self.encoders0 = nn.ModuleList(
            [
                EncoderLayerSANM(
                    self.input_size,
                    self.output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args0),
                    positionwise_layer(*positionwise_layer_args),
                    self.dropout_rate,
                )
                for i in range(1)
            ]
        )

        self.encoders = nn.ModuleList(
            [
                EncoderLayerSANM(
                    self.output_size,
                    self.output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    self.dropout_rate,
                )
                for i in range(self.num_blocks - 1)
            ]
        )

        self.tp_encoders = nn.ModuleList(
            [
                EncoderLayerSANM(
                    self.output_size,
                    self.output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    self.dropout_rate,
                )
                for i in range(self.tp_blocks)
            ]
        )

        self.after_norm = LayerNorm(self.output_size)

        self.tp_norm = LayerNorm(self.output_size)

    def forward(
        self,
        xs_pad: torch.Tensor,
    ):
        masks = None

        xs_pad *= self.output_size**0.5

        xs_pad = self.embed(xs_pad)

        # forward encoder1
        for layer_idx, encoder_layer in enumerate(self.encoders0):
            encoder_outs = encoder_layer(xs_pad, masks)
            xs_pad, masks = encoder_outs[0], encoder_outs[1]

        for layer_idx, encoder_layer in enumerate(self.encoders):
            encoder_outs = encoder_layer(xs_pad, masks)
            xs_pad, masks = encoder_outs[0], encoder_outs[1]

        xs_pad = self.after_norm(xs_pad)

        for layer_idx, encoder_layer in enumerate(self.tp_encoders):
            encoder_outs = encoder_layer(xs_pad, masks)
            xs_pad, masks = encoder_outs[0], encoder_outs[1]

        xs_pad = self.tp_norm(xs_pad)
        return xs_pad


class CTC(nn.Module):
    def __init__(
        self,
        odim: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0,
        ctc_type: str = "builtin",
        reduce: bool = True,
        ignore_nan_grad: bool = True,
        extra_linear: bool = True,
    ):
        super().__init__()
        eprojs = encoder_output_size
        self.dropout_rate = dropout_rate

        if extra_linear:
            self.ctc_lo = torch.nn.Linear(eprojs, odim)
        else:
            self.ctc_lo = None

    def softmax(self, hs_pad):
        """softmax of frame activations

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: softmax applied 3d tensor (B, Tmax, odim)
        """
        if self.ctc_lo is not None:
            return F.softmax(self.ctc_lo(hs_pad), dim=2)
        else:
            return F.softmax(hs_pad, dim=2)

    def log_softmax(self, hs_pad):
        """log_softmax of frame activations

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: log softmax applied 3d tensor (B, Tmax, odim)
        """
        if self.ctc_lo is not None:
            return F.log_softmax(self.ctc_lo(hs_pad), dim=2)
        else:
            return F.log_softmax(hs_pad, dim=2)

    def argmax(self, hs_pad):
        """argmax of frame activations

        Args:
            torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: argmax applied 2d tensor (B, Tmax)
        """
        if self.ctc_lo is not None:
            return torch.argmax(self.ctc_lo(hs_pad), dim=2)
        else:
            return torch.argmax(hs_pad, dim=2)


class SenseVoiceSmall(nn.Module):
    def __init__(self, neg_mean: torch.Tensor, inv_stddev: torch.Tensor):
        super().__init__()
        self.sos = 1
        self.eos = 2
        self.length_normalized_loss = True
        self.ignore_id = -1
        self.blank_id = 0
        self.input_size = 80 * 7
        self.vocab_size = 25055

        self.neg_mean = neg_mean.unsqueeze(0).unsqueeze(0)
        self.inv_stddev = inv_stddev.unsqueeze(0).unsqueeze(0)

        self.lid_dict = {
            "auto": 0,
            "zh": 3,
            "en": 4,
            "yue": 7,
            "ja": 11,
            "ko": 12,
            "nospeech": 13,
        }
        self.lid_int_dict = {
            24884: 3,
            24885: 4,
            24888: 7,
            24892: 11,
            24896: 12,
            24992: 13,
        }
        self.textnorm_dict = {"withitn": 14, "woitn": 15}
        self.textnorm_int_dict = {25016: 14, 25017: 15}

        self.emo_dict = {
            "unk": 25009,
            "happy": 25001,
            "sad": 25002,
            "angry": 25003,
            "neutral": 25004,
        }

        self.encoder = SenseVoiceEncoderSmall()
        self.ctc = CTC(
            odim=self.vocab_size,
            encoder_output_size=self.encoder.output_size,
        )
        self.embed = torch.nn.Embedding(
            7 + len(self.lid_dict) + len(self.textnorm_dict), self.input_size
        )

    def forward(self, x, prompt):
        input_query = self.embed(prompt).unsqueeze(0)

        # for export, we always assume x and self.neg_mean are on CPU
        x = (x + self.neg_mean) * self.inv_stddev
        x = torch.cat((input_query, x), dim=1)

        encoder_out = self.encoder(x)
        logits = self.ctc.ctc_lo(encoder_out)

        return logits
