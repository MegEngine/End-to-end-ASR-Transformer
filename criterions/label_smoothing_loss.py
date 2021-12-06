#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Label smoothing module."""

import megengine as mge
import megengine.module as M
import megengine.functional as F


class LabelSmoothingLoss(M.Module):
    def __init__(self, size, padding_idx, smoothing, normalize_length=True):
        """Construct an LabelSmoothingLoss object."""
        super(LabelSmoothingLoss, self).__init__()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        self.normalize_length = normalize_length

    def kl_div(self, p, q):
        return q * (F.log(q) - F.logsoftmax(p, axis=1))

    def forward(self, x, target):
        """Compute loss between x and target."""
        assert x.shape[2] == self.size
        batch_size = x.shape[0]

        x = x.reshape(-1, self.size)
        target = target.reshape(-1)

        # no grad:
        target_ = F.copy(target).detach()
        true_dist = F.zeros_like(x)
        true_dist[:] = self.smoothing / (self.size - 1)
        tmp_ignore = target_ == self.padding_idx  # (B,)
        total = len(target_) - tmp_ignore.sum().item()
        target_[tmp_ignore] = 0
        target_ = target_.reshape(-1, 1)
        confidence = F.ones_like(target_) * self.confidence
        confidence_ = confidence.detach()
        true_dist = F.scatter(true_dist, 1, target_, confidence_)

        kl = self.kl_div(x, true_dist)
        denom = total if self.normalize_length else batch_size
        kl[..., tmp_ignore] = 0
        return kl.sum() / denom
