#!/usr/bin/env python3
"""
Code in this file is copied and modified from
https://github.com/modelscope/FunASR
"""
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


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
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = torch.nn.LayerNorm(in_size)
        self.norm2 = torch.nn.LayerNorm(size)
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
        self,
        x,
        mask=None,
        cache=None,
        mask_shfit_chunk=None,
        mask_att_chunk_encoder=None,
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
        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if self.in_size == self.size:
            x = residual + self.dropout(
                self.self_attn(
                    x,
                    mask,
                    mask_shfit_chunk=mask_shfit_chunk,
                    mask_att_chunk_encoder=mask_att_chunk_encoder,
                )
            )
        else:
            x = self.dropout(
                self.self_attn(
                    x,
                    mask,
                    mask_shfit_chunk=mask_shfit_chunk,
                    mask_att_chunk_encoder=mask_att_chunk_encoder,
                )
            )

        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)

        x = residual + self.dropout(self.feed_forward(x))

        if not self.normalize_before:
            x = self.norm2(x)

        x = torch.clamp(x, -60000.0, 60000.0)

        return x, mask, cache, mask_shfit_chunk, mask_att_chunk_encoder


class MultiSequential(torch.nn.Sequential):
    """Multi-input multi-output torch.nn.Sequential."""

    def __init__(self, *args, layer_drop_rate=0.0):
        """Initialize MultiSequential with layer_drop.

        Args:
            layer_drop_rate (float): Probability of dropping out each fn (layer).

        """
        super().__init__(*args)
        self.layer_drop_rate = layer_drop_rate

    def forward(self, *args):
        """Repeat."""
        for idx, m in enumerate(self):
            args = m(*args)
        return args


def repeat(N, fn, layer_drop_rate=0.0):
    """Repeat module N times.

    Args:
        N (int): Number of repeat time.
        fn (Callable): Function to generate module.
        layer_drop_rate (float): Probability of dropping out each fn (layer).

    Returns:
        MultiSequential: Repeated model instance.

    """
    return MultiSequential(*[fn(n) for n in range(N)], layer_drop_rate=layer_drop_rate)


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
        """Construct an MultiHeadedAttention object."""
        super().__init__()

        assert lora_list is None

        assert n_feat % n_head == 0, (n_feat, n_head)

        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        # self.linear_q = nn.Linear(n_feat, n_feat)
        # self.linear_k = nn.Linear(n_feat, n_feat)
        # self.linear_v = nn.Linear(n_feat, n_feat)

        self.linear_out = nn.Linear(n_feat, n_feat)
        self.linear_q_k_v = nn.Linear(in_feat, n_feat * 3)
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

    def forward_fsmn(self, inputs, mask=None, mask_shfit_chunk=None):
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

    def forward_attention(self, value, scores, mask=None, mask_att_chunk_encoder=None):
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

    def forward(self, x, mask=None, mask_shfit_chunk=None, mask_att_chunk_encoder=None):
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


class SinusoidalPositionEncoder(torch.nn.Module):
    """ """

    def __init__(self, d_model=80, dropout_rate=0.1):
        super().__init__()
        pass

    def encode(
        self,
        positions: torch.Tensor = None,
        depth: int = None,
        dtype: torch.dtype = torch.float32,
    ):
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


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim, hidden_units, dropout_rate, activation=torch.nn.ReLU()):
        """Construct an PositionwiseFeedForward object."""
        super().__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x):
        """Forward function."""
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class SANMEncoder(nn.Module):
    """
    Author: Zhifu Gao, Shiliang Zhang, Ming Lei, Ian McLoughlin
    San-m: Memory equipped self-attention for end-to-end speech recognition
    https://arxiv.org/abs/2006.01713
    """

    def __init__(
        self,
        neg_mean: torch.Tensor,
        inv_stddev: torch.Tensor,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        pos_enc_class=SinusoidalPositionEncoder,
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        padding_idx: int = -1,
        interctc_layer_idx: List[int] = [],
        interctc_use_conditioning: bool = False,
        kernel_size: int = 11,
        sanm_shfit: int = 0,
        lora_list: List[str] = None,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        selfattention_layer_type: str = "sanm",
        tf2torch_tensor_name_prefix_torch: str = "encoder",
        tf2torch_tensor_name_prefix_tf: str = "seq2seq/encoder",
    ):
        super().__init__()
        self.neg_mean = neg_mean
        self.inv_stddev = inv_stddev
        self._output_size = output_size
        assert input_layer == "pe", input_layer

        self.embed = SinusoidalPositionEncoder()
        self.normalize_before = normalize_before

        assert positionwise_layer_type == "linear", positionwise_layer_type
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
        )

        assert selfattention_layer_type == "sanm", selfattention_layer_type

        encoder_selfattn_layer = MultiHeadedAttentionSANM
        encoder_selfattn_layer_args0 = (
            attention_heads,
            input_size,
            output_size,
            attention_dropout_rate,
            kernel_size,
            sanm_shfit,
            lora_list,
            lora_rank,
            lora_alpha,
            lora_dropout,
        )

        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            output_size,
            attention_dropout_rate,
            kernel_size,
            sanm_shfit,
            lora_list,
            lora_rank,
            lora_alpha,
            lora_dropout,
        )

        self.encoders0 = repeat(
            1,
            lambda lnum: EncoderLayerSANM(
                input_size,
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args0),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )

        self.encoders = repeat(
            num_blocks - 1,
            lambda lnum: EncoderLayerSANM(
                output_size,
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )

        if self.normalize_before:
            self.after_norm = torch.nn.LayerNorm(output_size)

        self.interctc_layer_idx = interctc_layer_idx

        assert len(interctc_layer_idx) == 0, len(interctc_layer_idx)
        self.interctc_use_conditioning = interctc_use_conditioning
        self.conditioning_layer = None
        self.dropout = nn.Dropout(dropout_rate)
        self.tf2torch_tensor_name_prefix_torch = tf2torch_tensor_name_prefix_torch
        self.tf2torch_tensor_name_prefix_tf = tf2torch_tensor_name_prefix_tf

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
    ) -> torch.Tensor:
        """Embed positions in tensor.

        Args:
            xs_pad: input tensor (B, L, D)
        Returns:
            position embedded tensor and mask
        """
        print("in xs_pad.shape", xs_pad.shape)
        xs_pad = (xs_pad + self.neg_mean) * self.inv_stddev
        masks = None
        xs_pad = xs_pad * self.output_size() ** 0.5

        xs_pad = self.embed(xs_pad)

        # xs_pad = self.dropout(xs_pad)
        encoder_outs = self.encoders0(xs_pad, masks)
        xs_pad, masks = encoder_outs[0], encoder_outs[1]
        encoder_outs = self.encoders(xs_pad, masks)
        xs_pad, masks = encoder_outs[0], encoder_outs[1]

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        print("out xs_pad.shape", xs_pad.shape)
        return xs_pad


def _pre_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    """Perform pre-hook in load_state_dict for backward compatibility.

    Note:
        We saved self.pe until v.0.5.2 but we have omitted it later.
        Therefore, we remove the item "pe" from `state_dict` for backward compatibility.

    """
    k = prefix + "pe"
    if k in state_dict:
        state_dict.pop(k)


class DecoderLayerSANM(torch.nn.Module):
    """Single decoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        src_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)


    """

    def __init__(
        self,
        size,
        self_attn,
        src_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
    ):
        """Construct an DecoderLayer object."""
        super(DecoderLayerSANM, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = torch.nn.LayerNorm(size)
        if self_attn is not None:
            self.norm2 = torch.nn.LayerNorm(size)
        if src_attn is not None:
            self.norm3 = torch.nn.LayerNorm(size)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear1 = torch.nn.Linear(size + size, size)
            self.concat_linear2 = torch.nn.Linear(size + size, size)
        self.reserve_attn = False

    def forward(self, tgt, tgt_mask, memory, memory_mask=None, cache=None):
        """Compute decoded features.

        Args:
            tgt (torch.Tensor): Input tensor (#batch, maxlen_out, size).
            tgt_mask (torch.Tensor): Mask for input tensor (#batch, maxlen_out).
            memory (torch.Tensor): Encoded memory, float32 (#batch, maxlen_in, size).
            memory_mask (torch.Tensor): Encoded memory mask (#batch, maxlen_in).
            cache (List[torch.Tensor]): List of cached tensors.
                Each tensor shape should be (#batch, maxlen_out - 1, size).

        Returns:
            torch.Tensor: Output tensor(#batch, maxlen_out, size).
            torch.Tensor: Mask for output tensor (#batch, maxlen_out).
            torch.Tensor: Encoded memory (#batch, maxlen_in, size).
            torch.Tensor: Encoded memory mask (#batch, maxlen_in).

        """
        # tgt = self.dropout(tgt)
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        tgt = self.feed_forward(tgt)

        x = tgt
        if self.self_attn:
            if self.normalize_before:
                tgt = self.norm2(tgt)
            x, _ = self.self_attn(tgt, tgt_mask)
            x = residual + self.dropout(x)

        if self.src_attn is not None:
            residual = x
            if self.normalize_before:
                x = self.norm3(x)

            x_src_attn = self.src_attn(x, memory, memory_mask, ret_attn=False)
            x = residual + self.dropout(x_src_attn)
            # x = residual + self.dropout(self.src_attn(x, memory, memory_mask))

        return x, tgt_mask, memory, memory_mask, cache


class MultiHeadedAttentionSANMDecoder(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_feat, dropout_rate, kernel_size, sanm_shfit=0):
        """Construct an MultiHeadedAttention object."""
        super().__init__()

        self.dropout = nn.Dropout(p=dropout_rate)

        self.fsmn_block = nn.Conv1d(
            n_feat, n_feat, kernel_size, stride=1, padding=0, groups=n_feat, bias=False
        )
        # padding
        # padding
        left_padding = (kernel_size - 1) // 2
        if sanm_shfit > 0:
            left_padding = left_padding + sanm_shfit
        right_padding = kernel_size - 1 - left_padding
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)
        self.kernel_size = kernel_size

    def forward(self, inputs, mask, cache=None, mask_shfit_chunk=None):
        """
        :param x: (#batch, time1, size).
        :param mask: Mask tensor (#batch, 1, time)
        :return:
        """
        # print("in fsmn, inputs", inputs.size())
        b, t, d = inputs.size()
        # logging.info(
        #     "mask: {}".format(mask.size()))
        if mask is not None:
            mask = torch.reshape(mask, (b, -1, 1))
            # logging.info("in fsmn, mask: {}, {}".format(mask.size(), mask[0:100:50, :, :]))
            if mask_shfit_chunk is not None:
                # logging.info("in fsmn, mask_fsmn: {}, {}".format(mask_shfit_chunk.size(), mask_shfit_chunk[0:100:50, :, :]))
                mask = mask * mask_shfit_chunk
            # logging.info("in fsmn, mask_after_fsmn: {}, {}".format(mask.size(), mask[0:100:50, :, :]))
            # print("in fsmn, mask", mask.size())
            # print("in fsmn, inputs", inputs.size())
            inputs = inputs * mask

        x = inputs.transpose(1, 2)
        b, d, t = x.size()
        if cache is None:
            # print("in fsmn, cache is None, x", x.size())

            x = self.pad_fn(x)
            if not self.training:
                cache = x
        else:
            # print("in fsmn, cache is not None, x", x.size())
            # x = torch.cat((x, cache), dim=2)[:, :, :-1]
            # if t < self.kernel_size:
            #     x = self.pad_fn(x)
            x = torch.cat((cache[:, :, 1:], x), dim=2)
            x = x[:, :, -(self.kernel_size + t - 1) :]
            # print("in fsmn, cache is not None, x_cat", x.size())
            cache = x
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        # print("in fsmn, fsmn_out", x.size())
        if x.size(1) != inputs.size(1):
            inputs = inputs[:, -1, :]

        x = x + inputs
        x = self.dropout(x)
        if mask is not None:
            x = x * mask
        return x, cache


class MultiHeadedAttentionCrossAtt(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate,
        lora_list=None,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
        encoder_output_size=None,
    ):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k_v = nn.Linear(
            n_feat if encoder_output_size is None else encoder_output_size,
            n_feat * 2,
        )
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, x, memory):
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

        # print("in forward_qkv, x", x.size())
        b = x.size(0)
        q = self.linear_q(x)
        q_h = torch.reshape(q, (b, -1, self.h, self.d_k)).transpose(
            1, 2
        )  # (batch, head, time1, d_k)

        k_v = self.linear_k_v(memory)
        k, v = torch.split(k_v, int(self.h * self.d_k), dim=-1)
        k_h = torch.reshape(k, (b, -1, self.h, self.d_k)).transpose(
            1, 2
        )  # (batch, head, time2, d_k)
        v_h = torch.reshape(v, (b, -1, self.h, self.d_k)).transpose(
            1, 2
        )  # (batch, head, time2, d_k)

        return q_h, k_h, v_h

    def forward_attention(self, value, scores, mask, ret_attn=False):
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
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = -float(
                "inf"
            )  # float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            # logging.info(
            #     "scores: {}, mask_size: {}".format(scores.size(), mask.size()))
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
        if ret_attn:
            return self.linear_out(x), attn  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, x, memory, memory_mask, ret_attn=False):
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
        q_h, k_h, v_h = self.forward_qkv(x, memory)
        q_h = q_h * self.d_k ** (-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        return self.forward_attention(v_h, scores, memory_mask, ret_attn=ret_attn)


class PositionwiseFeedForwardDecoderSANM(torch.nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(
        self, idim, hidden_units, dropout_rate, adim=None, activation=torch.nn.ReLU()
    ):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForwardDecoderSANM, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(
            hidden_units, idim if adim is None else adim, bias=False
        )
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation
        self.norm = torch.nn.LayerNorm(hidden_units)

    def forward(self, x):
        """Forward function."""
        return self.w_2(self.norm(self.dropout(self.activation(self.w_1(x)))))


class ParaformerSANMDecoder(torch.nn.Module):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2006.01713
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        wo_input_layer: bool = False,
        pos_enc_class="PositionalEncoding",
        normalize_before: bool = True,
        concat_after: bool = False,
        att_layer_num: int = 6,
        kernel_size: int = 21,
        sanm_shfit: int = 0,
        lora_list: List[str] = None,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        chunk_multiply_factor: tuple = (1,),
        tf2torch_tensor_name_prefix_torch: str = "decoder",
        tf2torch_tensor_name_prefix_tf: str = "seq2seq/decoder",
    ):
        super().__init__()

        attention_dim = encoder_output_size

        assert wo_input_layer is False
        assert input_layer == "embed", input_layer

        # Note: self.embed is not used
        self.embed = torch.nn.Sequential(
            torch.nn.Embedding(vocab_size, attention_dim),
            # pos_enc_class(attention_dim, positional_dropout_rate),
        )

        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = torch.nn.LayerNorm(attention_dim)
        if use_output_layer:
            self.output_layer = torch.nn.Linear(attention_dim, vocab_size)
        else:
            self.output_layer = None

        self.att_layer_num = att_layer_num
        self.num_blocks = num_blocks
        if sanm_shfit is None:
            sanm_shfit = (kernel_size - 1) // 2
        self.decoders = repeat(
            att_layer_num,
            lambda lnum: DecoderLayerSANM(
                attention_dim,
                MultiHeadedAttentionSANMDecoder(
                    attention_dim,
                    self_attention_dropout_rate,
                    kernel_size,
                    sanm_shfit=sanm_shfit,
                ),
                MultiHeadedAttentionCrossAtt(
                    attention_heads,
                    attention_dim,
                    src_attention_dropout_rate,
                    lora_list,
                    lora_rank,
                    lora_alpha,
                    lora_dropout,
                ),
                PositionwiseFeedForwardDecoderSANM(
                    attention_dim, linear_units, dropout_rate
                ),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        if num_blocks - att_layer_num <= 0:
            self.decoders2 = None
        else:
            self.decoders2 = repeat(
                num_blocks - att_layer_num,
                lambda lnum: DecoderLayerSANM(
                    attention_dim,
                    MultiHeadedAttentionSANMDecoder(
                        attention_dim,
                        self_attention_dropout_rate,
                        kernel_size,
                        sanm_shfit=0,
                    ),
                    None,
                    PositionwiseFeedForwardDecoderSANM(
                        attention_dim, linear_units, dropout_rate
                    ),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ),
            )

        self.decoders3 = repeat(
            1,
            lambda lnum: DecoderLayerSANM(
                attention_dim,
                None,
                None,
                PositionwiseFeedForwardDecoderSANM(
                    attention_dim, linear_units, dropout_rate
                ),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        self.tf2torch_tensor_name_prefix_torch = tf2torch_tensor_name_prefix_torch
        self.tf2torch_tensor_name_prefix_tf = tf2torch_tensor_name_prefix_tf
        self.chunk_multiply_factor = chunk_multiply_factor

    def forward(
        self,
        hs_pad: torch.Tensor,
        ys_in_pad: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        chunk_mask: torch.Tensor = None,
        return_hidden: bool = False,
        return_both: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward decoder.

        Args:
            hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
            ys_in_pad:
                input token ids, int64 (batch, maxlen_out)
                if input_layer == "embed"
                input tensor (batch, maxlen_out, #mels) in the other cases
        Returns:
            (tuple): tuple containing:

            x: decoded token score before softmax (batch, maxlen_out, token)
                if use_output_layer is True,
        """
        tgt = ys_in_pad

        memory = hs_pad
        memory_mask = None

        x = tgt
        x, tgt_mask, memory, memory_mask, _ = self.decoders(
            x, tgt_mask, memory, memory_mask
        )
        if self.decoders2 is not None:
            x, tgt_mask, memory, memory_mask, _ = self.decoders2(
                x, tgt_mask, memory, memory_mask
            )
        x, tgt_mask, memory, memory_mask, _ = self.decoders3(
            x, tgt_mask, memory, memory_mask
        )
        if self.normalize_before:
            hidden = self.after_norm(x)

        print("hidden", hidden.shape)
        print("self.output_layer", self.output_layer)
        x = self.output_layer(hidden)
        print("x", x.shape)
        return x


def cif_wo_hidden_v1(alphas, threshold, return_fire_idxs=False):
    batch_size, len_time = alphas.size()
    device = alphas.device
    dtype = alphas.dtype

    threshold = torch.tensor([threshold], dtype=alphas.dtype).to(alphas.device)

    fires = torch.zeros(batch_size, len_time, dtype=dtype, device=device)

    # prefix_sum = torch.cumsum(alphas, dim=1)
    prefix_sum = torch.cumsum(alphas, dim=1, dtype=torch.float64).to(
        torch.float32
    )  # cumsum precision degradation cause wrong result in extreme
    prefix_sum_floor = torch.floor(prefix_sum)
    dislocation_prefix_sum = torch.roll(prefix_sum, 1, dims=1)
    dislocation_prefix_sum_floor = torch.floor(dislocation_prefix_sum)

    dislocation_prefix_sum_floor[:, 0] = 0
    dislocation_diff = prefix_sum_floor - dislocation_prefix_sum_floor

    fire_idxs = dislocation_diff > 0
    fires[fire_idxs] = 1
    fires = fires + prefix_sum - prefix_sum_floor
    if return_fire_idxs:
        return fires, fire_idxs
    return fires


def cif_v1(hidden, alphas, threshold):
    fires, fire_idxs = cif_wo_hidden_v1(alphas, threshold, return_fire_idxs=True)

    device = hidden.device
    dtype = hidden.dtype
    batch_size, len_time, hidden_size = hidden.size()
    # frames = torch.zeros(batch_size, len_time, hidden_size, dtype=dtype, device=device)
    # prefix_sum_hidden = torch.cumsum(alphas.unsqueeze(-1).tile((1, 1, hidden_size)) * hidden, dim=1)
    frames = torch.zeros(batch_size, len_time, hidden_size, dtype=dtype, device=device)
    prefix_sum_hidden = torch.cumsum(
        alphas.unsqueeze(-1).repeat((1, 1, hidden_size)) * hidden, dim=1
    )

    frames = prefix_sum_hidden[fire_idxs]
    shift_frames = torch.roll(frames, 1, dims=0)

    batch_len = fire_idxs.sum(1)
    batch_idxs = torch.cumsum(batch_len, dim=0)
    shift_batch_idxs = torch.roll(batch_idxs, 1, dims=0)
    shift_batch_idxs[0] = 0
    shift_frames[shift_batch_idxs] = 0

    remains = fires - torch.floor(fires)
    # remain_frames = remains[fire_idxs].unsqueeze(-1).tile((1, hidden_size)) * hidden[fire_idxs]
    remain_frames = (
        remains[fire_idxs].unsqueeze(-1).repeat((1, hidden_size)) * hidden[fire_idxs]
    )

    shift_remain_frames = torch.roll(remain_frames, 1, dims=0)
    shift_remain_frames[shift_batch_idxs] = 0

    frames = frames - shift_frames + shift_remain_frames - remain_frames

    # max_label_len = batch_len.max()
    max_label_len = (
        torch.round(alphas.sum(-1)).int().max()
    )  # torch.round to calculate the max length

    # frame_fires = torch.zeros(batch_size, max_label_len, hidden_size, dtype=dtype, device=device)
    frame_fires = torch.zeros(
        batch_size, max_label_len, hidden_size, dtype=dtype, device=device
    )
    indices = torch.arange(max_label_len, device=device).expand(batch_size, -1)
    frame_fires_idxs = indices < batch_len.unsqueeze(1)
    frame_fires[frame_fires_idxs] = frames
    return frame_fires, fires


class CifPredictorV2(torch.nn.Module):
    def __init__(
        self,
        idim,
        l_order,
        r_order,
        threshold=1.0,
        dropout=0.1,
        smooth_factor=1.0,
        noise_threshold=0,
        tail_threshold=0.0,
        tf2torch_tensor_name_prefix_torch="predictor",
        tf2torch_tensor_name_prefix_tf="seq2seq/cif",
        tail_mask=True,
    ):
        super().__init__()

        self.pad = torch.nn.ConstantPad1d((l_order, r_order), 0)
        self.cif_conv1d = torch.nn.Conv1d(idim, idim, l_order + r_order + 1)
        self.cif_output = torch.nn.Linear(idim, 1)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.threshold = threshold
        self.smooth_factor = smooth_factor
        self.noise_threshold = noise_threshold
        self.tail_threshold = tail_threshold
        self.tf2torch_tensor_name_prefix_torch = tf2torch_tensor_name_prefix_torch
        self.tf2torch_tensor_name_prefix_tf = tf2torch_tensor_name_prefix_tf
        self.tail_mask = tail_mask

    def forward(
        self,
        hidden,
        target_label=None,
        mask=None,
        ignore_id=-1,
        mask_chunk_predictor=None,
        target_label_length=None,
    ):
        h = hidden
        context = h.transpose(1, 2)
        queries = self.pad(context)
        output = torch.relu(self.cif_conv1d(queries))
        output = output.transpose(1, 2)

        output = self.cif_output(output)
        alphas = torch.sigmoid(output)
        alphas = torch.nn.functional.relu(
            alphas * self.smooth_factor - self.noise_threshold
        )
        if mask is not None:
            mask = mask.transpose(-1, -2).float()
            alphas = alphas * mask
        if mask_chunk_predictor is not None:
            alphas = alphas * mask_chunk_predictor

        alphas = alphas.squeeze(-1)
        if mask is not None:
            mask = mask.squeeze(-1)

        if target_label_length is not None:
            target_length = target_label_length.squeeze(-1)
        elif target_label is not None:
            target_length = (target_label != ignore_id).float().sum(-1)
        else:
            target_length = None
        token_num = alphas.sum(-1)
        if target_length is not None:
            alphas *= (target_length / token_num)[:, None].repeat(1, alphas.size(1))
        elif self.tail_threshold > 0.0:
            if self.tail_mask:
                hidden, alphas, token_num = self.tail_process_fn(
                    hidden, alphas, token_num, mask=mask
                )
            else:
                hidden, alphas, token_num = self.tail_process_fn(
                    hidden, alphas, token_num, mask=None
                )

        acoustic_embeds, cif_peak = cif_v1(hidden, alphas, self.threshold)
        if target_length is None and self.tail_threshold > 0.0:
            token_num_int = torch.max(token_num).type(torch.int32).item()
            acoustic_embeds = acoustic_embeds[:, :token_num_int, :]

        return acoustic_embeds, token_num, alphas, cif_peak

    def tail_process_fn(self, hidden, alphas, token_num=None, mask=None):
        b, t, d = hidden.size()
        tail_threshold = self.tail_threshold
        if mask is not None:
            zeros_t = torch.zeros((b, 1), dtype=torch.float32, device=alphas.device)
            ones_t = torch.ones_like(zeros_t)
            mask_1 = torch.cat([mask, zeros_t], dim=1)
            mask_2 = torch.cat([ones_t, mask], dim=1)
            mask = mask_2 - mask_1
            tail_threshold = mask * tail_threshold
            alphas = torch.cat([alphas, zeros_t], dim=1)
            alphas = torch.add(alphas, tail_threshold)
        else:
            tail_threshold = torch.tensor([tail_threshold], dtype=alphas.dtype).to(
                alphas.device
            )
            tail_threshold = torch.reshape(tail_threshold, (1, 1))
            if b > 1:
                alphas = torch.cat([alphas, tail_threshold.repeat(b, 1)], dim=1)
            else:
                alphas = torch.cat([alphas, tail_threshold], dim=1)
        zeros = torch.zeros((b, 1, d), dtype=hidden.dtype).to(hidden.device)
        hidden = torch.cat([hidden, zeros], dim=1)
        token_num = alphas.sum(dim=-1)
        token_num_floor = torch.floor(token_num)

        return hidden, alphas, token_num_floor


class Paraformer(torch.nn.Module):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2206.08317
    """

    def __init__(
        self,
        neg_mean: torch.Tensor,
        inv_stddev: torch.Tensor,
        input_size: int,
        vocab_size: int,
        ignore_id=-1,
        encoder_conf: Optional[Dict] = None,
        decoder_conf: Optional[Dict] = None,
        predictor_conf: Optional[Dict] = None,
    ):
        super().__init__()

        self.ignore_id = ignore_id
        self.encoder = SANMEncoder(
            neg_mean, inv_stddev, input_size=input_size, **encoder_conf
        )
        encoder_output_size = self.encoder.output_size()

        self.decoder = ParaformerSANMDecoder(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            **decoder_conf,
        )
        self.predictor = CifPredictorV2(**predictor_conf)

    def forward(self, x):
        """
        Args:
          x: (N, T, C)
        """
        encoder_out = self.encoder(x)

        encoder_out_mask = None

        pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = self.predictor(
            encoder_out, None, encoder_out_mask, ignore_id=self.ignore_id
        )
        # pre_acoustic_embeds: (N, num_tokens, C)
        # pre_token_length: [num_tokens,]
        # alphas: (N, T)
        # pre_peak_index: (N, T)

        pre_token_length = pre_token_length.round().long()
        if torch.max(pre_token_length) < 1:
            return []

        decoder_outs = self.decoder(encoder_out, pre_acoustic_embeds)
        # decoder_outs: (N, num_tokens, vocab_size)
        return decoder_outs, pre_token_length


@torch.no_grad()
def test():
    import yaml

    with open("./config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(config["encoder_conf"])

    neg_mean = torch.rand(560)
    inv_stddev = torch.rand(560)

    m = Paraformer(
        neg_mean=neg_mean,
        inv_stddev=inv_stddev,
        input_size=560,
        vocab_size=8404,
        encoder_conf=config["encoder_conf"],
        decoder_conf=config["decoder_conf"],
        predictor_conf=config["predictor_conf"],
    )
    m.eval()
    print(m.decoder)

    state_dict = torch.load("./model_state_dict.pt", map_location="cpu")["state_dict"]
    m.load_state_dict(state_dict)
    del state_dict
    print(m)


if __name__ == "__main__":
    test()
