from torch import nn

from torch_model import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """
    This class is copied and modified from
    https://github.com/modelscope/FunASR/blob/main/funasr/models/transformer/encoder.py
    """

    def __init__(
        self,
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
        self.norm1 = nn.LayerNorm(size, eps=1e-12)
        self.norm2 = nn.LayerNorm(size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate

    def forward(self, x, mask, cache=None):
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

        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]

        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x_q, x, x, mask)), dim=-1)
            x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
        else:
            x = residual + stoch_layer_coeff * self.dropout(
                self.self_attn(x_q, x, x, mask)
            )
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + stoch_layer_coeff * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x, mask


class Transformer(nn.Module):
    # This class is copied and modified from
    # https://github.com/modelscope/FunASR/blob/main/funasr/models/llm_asr/adaptor.py
    def __init__(
        self,
        downsample_rate=1,
        encoder_dim=512,
        llm_dim=512,
        ffn_dim: int = 2048,
        n_layer: int = 5,
        **kwargs
    ):
        super().__init__()
        self.k = downsample_rate
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.linear1 = nn.Linear(self.encoder_dim * self.k, ffn_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(ffn_dim, self.llm_dim)

        self.blocks = None
        if n_layer > 0:
            self.blocks = nn.ModuleList(
                [
                    EncoderLayer(
                        llm_dim,
                        MultiHeadedAttention(
                            kwargs.get("attention_heads", 8),
                            llm_dim,
                            kwargs.get("attention_dropout_rate", 0.0),
                        ),
                        PositionwiseFeedForward(
                            llm_dim,
                            llm_dim // 4,
                            kwargs.get("dropout_rate", 0.0),
                        ),
                        kwargs.get("dropout_rate", 0.0),
                    )
                    for i in range(n_layer)
                ]
            )

    def forward(self, x, ilens=None):
        batch_size, seq_len, dim = x.size()
        chunk_num = (seq_len - 1) // self.k + 1
        pad_num = chunk_num * self.k - seq_len
        x = F.pad(x, (0, 0, 0, pad_num, 0, 0), value=0.0)
        seq_len = x.size(1)

        x = x.contiguous()
        x = x.view(batch_size, chunk_num, dim * self.k)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        olens = None
        olens = (ilens - 1) // self.k + 1
        #  masks = (~make_pad_mask(olens)[:, None, :]).to(x.device)
        masks = None

        if self.blocks is not None:
            for layer, block in enumerate(self.blocks):
                x, masks = block(x, masks)
        return x, olens
