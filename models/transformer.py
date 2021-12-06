from argparse import Namespace
import logging
import math
import numpy as np
import megengine.module as M
import megengine.functional as F
import megengine as mge
from utils.net_utils import make_non_pad_mask
from .mask import subsequent_mask
from .mask import target_mask
import hparams as hp
from .encoder import Encoder
from .decoder import Decoder


class Model(M.Module):
    def __init__(self, idim, odim):
        M.Module.__init__(self)
        self.encoder = Encoder(
            idim=idim,
            attention_dim=hp.adim,
            attention_heads=hp.aheads,
            linear_units=hp.eunits,
            num_blocks=hp.elayers,
            dropout_rate=hp.dropout_rate,
            positional_dropout_rate=hp.dropout_rate,
            attention_dropout_rate=hp.dropout_rate,
        )
        self.decoder = Decoder(
            odim=odim,
            attention_dim=hp.adim,
            attention_heads=hp.aheads,
            linear_units=hp.dunits,
            num_blocks=hp.dlayers,
            dropout_rate=hp.dropout_rate,
            positional_dropout_rate=hp.dropout_rate,
            self_attention_dropout_rate=hp.dropout_rate,
            src_attention_dropout_rate=hp.dropout_rate,
        )
        self.ignore_id = 0
        self.sos = 2
        self.eos = 3

    def forward(self, xs_pad, xlens, ys_in_pad, ylens, evaluate=False):
        if evaluate:
            return self.evaluate(xs_pad, xlens, ys_in_pad, ylens)
        src_mask = make_non_pad_mask(xlens, xs_pad.shape[1])
        src_mask = src_mask.reshape(src_mask.shape[0], 1, src_mask.shape[1])

        hs_pad, hs_mask = self.encoder(xs_pad, src_mask)

        ys_mask = target_mask(ys_in_pad, self.ignore_id)
        pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)

        return hs_pad, hs_mask, pred_pad, pred_mask

    def evaluate(self, xs_pad, xlens, ys_in_pad, ylens):
        self.eval()
        src_mask = make_non_pad_mask(xlens, xs_pad.shape[1])
        src_mask = src_mask.reshape(src_mask.shape[0], 1, src_mask.shape[1])
        hs_pad, hs_mask = self.encoder(xs_pad, src_mask)

        ys = mge.Tensor([self.sos] * xs_pad.shape[0]).reshape(-1, 1)
        catch = None

        for i in range(ys_in_pad.shape[1]):
            ys_mask = target_mask(ys, self.ignore_id)
            local_att_scores, catch = self.decoder.forward_one_step(
                ys, ys_mask, hs_pad, hs_mask, catch
            )
            local_best_scores, local_best_ids = F.topk(
                local_att_scores.detach().reshape(
                    local_att_scores.shape[0], local_att_scores.shape[2]
                ),
                1,
                descending=True,
            )
            ys = F.concat([ys, local_best_ids], axis=1)
        return ys[:, 1:]
