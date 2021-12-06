import math
import numpy
import megengine as mge
import megengine.module as M
import megengine.functional as F

from .layer_norm import LayerNorm


class DecoderLayer(M.Module):
    """Single decoder layer module."""

    def __init__(
        self,
        size,
        self_attn,
        src_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
    ):
        """Construct an DecoderLayer object."""
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.norm3 = LayerNorm(size)
        self.dropout = M.dropout.Dropout(dropout_rate)
        self.normalize_before = normalize_before

    def forward(self, tgt, tgt_mask, memory, memory_mask, cache=None):
        """Compute decoded features.

        Args:
            tgt (megengine.Tensor): decoded previous target features (batch, max_time_out, size)
            tgt_mask (megengine.Tensor): mask for x (batch, max_time_out)
            memory (megengine.Tensor): encoded source features (batch, max_time_in, size)
            memory_mask (megengine.Tensor): mask for memory (batch, max_time_in)
            cache (megengine.Tensor): cached output (batch, max_time_out-1, size)

        """
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert cache.shape == (
                tgt.shape[0],
                tgt.shape[1] - 1,
                self.size,
            ), f"{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.size)}"
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = None
            if tgt_mask is not None:
                tgt_q_mask = tgt_mask[:, -1:, :]

        x = residual + self.dropout(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask))

        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)

        x = residual + self.dropout(self.src_attn(x, memory, memory, memory_mask))
        if not self.normalize_before:
            x = self.norm2(x)

        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)

        if cache is not None:
            x = F.concat([cache, x], axis=1)

        return x, tgt_mask, memory, memory_mask
