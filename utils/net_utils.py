import logging
import numpy as np
import megengine.module as M
import megengine.functional as F
import megengine as mge


def pad_list(xs, pad_value):
    """Perform padding for the list of tensors."""
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]

    return pad


def mask_by_length(xs, lengths, fill=0):
    """Mask tensor according to length."""
    assert xs.size(0) == len(lengths)
    ret = xs.data.new(*xs.size()).fill_(fill)
    for i, l in enumerate(lengths):
        ret[i, :l] = xs[i, :l]
    return ret


def make_pad_mask(lengths, maxlen=None):
    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if maxlen is None:
        maxlen = int(max(lengths))
    seq_range = mge.Tensor(F.arange(0, maxlen, dtype="int32"))
    seq_range_expand = F.broadcast_to(
        F.reshape(seq_range, (1, seq_range.shape[0])), (bs, maxlen)
    )
    seq_length_expand = mge.Tensor(lengths).reshape(-1, 1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def make_non_pad_mask(lengths, maxlen=None):
    return ~make_pad_mask(lengths, maxlen)
