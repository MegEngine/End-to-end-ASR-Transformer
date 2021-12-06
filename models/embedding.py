import math
import megengine.module as M
import megengine.functional as F
import megengine as mge


class PositionalEncoding(M.Module):
    """Positional encoding.

    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length

    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = M.dropout.Dropout(dropout_rate)
        self.pe = mge.Tensor(0.0)
        self.extend_pe(F.tensor.zeros([1, max_len]))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if len(self.pe.shape):
            if self.pe.shape[1] >= x.shape[1]:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = F.tensor.zeros([x.shape[1], self.d_model])
        position = mge.Tensor(F.arange(0, x.shape[1], dtype="float32")).reshape(
            x.shape[1], -1
        )
        div_term = F.exp(
            mge.Tensor(F.arange(0, self.d_model, 2, dtype="float32"))
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = F.sin(position * div_term)
        pe[:, 1::2] = F.cos(position * div_term)
        h, w = pe.shape
        pe = pe.reshape(-1, h, w)
        self.pe[...] = pe.to(device=x.device)

    def forward(self, x: mge.Tensor):
        """Add positional encoding.

        Args:
            x (megengine.Tensor): Input. Its shape is (batch, time, ...)

        Returns:
            megengine.Tensor: Encoded tensor. Its shape is (batch, time, ...)

        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, : x.shape[1]]
        return self.dropout(x)
