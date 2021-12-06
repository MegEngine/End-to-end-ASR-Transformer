import megengine as mge
import numpy as np


def subsequent_mask(size, device="cpu"):
    ret = mge.Tensor(np.tril(np.ones([size, size], dtype=bool)))
    return ret


def target_mask(ys_in_pad, ignore_id):
    """Create mask for decoder self-attention."""
    ys_mask = ys_in_pad != ignore_id
    m = subsequent_mask(ys_mask.shape[-1])
    m = m.reshape(1, m.shape[0], m.shape[1])
    ys_mask = ys_mask.reshape(ys_mask.shape[0], 1, ys_mask.shape[1])
    return ys_mask & m
