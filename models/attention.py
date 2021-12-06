import math
import numpy
import megengine as mge
import megengine.module as M
import megengine.functional as F


class MultiHeadedAttention(M.Module):
    """Multi-Head Attention layer.

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate

    """

    def __init__(self, n_head, n_feat, dropout_rate, flag="encoder"):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = M.Linear(n_feat, n_feat)
        self.linear_k = M.Linear(n_feat, n_feat)
        self.linear_v = M.Linear(n_feat, n_feat)
        self.linear_out = M.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = M.dropout.Dropout(dropout_rate)
        self.flag = flag

    def forward(self, query, key, value, mask):
        """Compute 'Scaled Dot Product Attention'.

        :param megengine.Tensor query: (batch, time1, size)
        :param megengine.Tensor key: (batch, time2, size)
        :param megengine.Tensor value: (batch, time2, size)
        :param megengine.Tensor mask: (batch, time1, time2)
        :param megengine.nn.Dropout dropout:
        :return megengine.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = query.shape[0]
        q = self.linear_q(query).reshape(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).reshape(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).reshape(n_batch, -1, self.h, self.d_k)
        q = F.transpose(q, (0, 2, 1, 3))
        k = F.transpose(k, (0, 2, 1, 3))
        v = F.transpose(v, (0, 2, 1, 3))

        scores = F.matmul(q, F.transpose(k, (0, 1, 3, 2))) / math.sqrt(
            self.d_k
        )  # (batch, head, time1, time2)
        if mask is not None:
            n, h, f = mask.shape
            if self.flag == "encoder":
                mask = mask.reshape(n, f)
                mask = ~mask
                min_value = float(
                    numpy.finfo(mge.tensor(0, dtype=scores.dtype).numpy().dtype).min
                )
                scores = F.transpose(scores, (2, 1, 0, 3))
                scores[..., mask] = min_value
                self.attn = F.softmax(scores, axis=-1)
                self.attn[..., mask] = 0.0
                self.attn = F.transpose(self.attn, (2, 1, 0, 3))
            else:
                if h == 1:
                    mask = mask.reshape(n, f)
                    mask = ~mask
                    min_value = float(
                        numpy.finfo(mge.tensor(0, dtype=scores.dtype).numpy().dtype).min
                    )
                    scores = F.transpose(scores, (2, 1, 0, 3))
                    scores[..., mask] = min_value
                    self.attn = F.softmax(scores, axis=-1)
                    self.attn[..., mask] = 0.0
                    self.attn = F.transpose(self.attn, (2, 1, 0, 3))
                else:
                    mask = ~mask
                    min_value = float(
                        numpy.finfo(mge.tensor(0, dtype=scores.dtype).numpy().dtype).min
                    )
                    scores = F.transpose(scores, (1, 0, 2, 3))
                    scores[..., mask] = min_value
                    self.attn = F.softmax(scores, axis=-1)
                    self.attn[..., mask] = 0.0
                    self.attn = F.transpose(self.attn, (1, 0, 2, 3))
        else:
            self.attn = F.softmax(scores, axis=-1)  # (batch, head, time1, time2)
        p_attn = self.dropout(self.attn)
        x = F.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = F.transpose(x, (0, 2, 1, 3)).reshape(
            n_batch, -1, self.h * self.d_k
        )  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)
