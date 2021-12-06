import megengine.module as M
import megengine.functional as F
from .attention import MultiHeadedAttention
from .decoder_layer import DecoderLayer
from .embedding import PositionalEncoding
from .layer_norm import LayerNorm
from .mask import subsequent_mask
from .positionwise_feed_forward import PositionwiseFeedForward
from .repeat import repeat


class Decoder(M.Module):
    """Transfomer decoder module."""

    def __init__(
        self,
        odim,
        attention_dim=256,
        attention_heads=4,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        self_attention_dropout_rate=0.0,
        src_attention_dropout_rate=0.0,
        input_layer="embed",
        use_output_layer=True,
        pos_enc_class=PositionalEncoding,
        normalize_before=True,
        concat_after=False,
    ):
        """Construct an Decoder object."""
        M.Module.__init__(self)
        if input_layer == "embed":
            self.embed = M.Sequential(
                M.Embedding(odim, attention_dim),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        self.normalize_before = normalize_before
        self.decoders = repeat(
            num_blocks,
            lambda: DecoderLayer(
                attention_dim,
                MultiHeadedAttention(
                    attention_heads,
                    attention_dim,
                    self_attention_dropout_rate,
                    flag="decoder",
                ),
                MultiHeadedAttention(
                    attention_heads,
                    attention_dim,
                    src_attention_dropout_rate,
                    flag="decoder",
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
        if use_output_layer:
            self.output_layer = M.Linear(attention_dim, odim)
        else:
            self.output_layer = None

    def forward(self, tgt, tgt_mask, memory, memory_mask):
        """Forward decoder."""
        x = self.embed(tgt)
        x, tgt_mask, memory, memory_mask = self.decoders(
            x, tgt_mask, memory, memory_mask
        )
        if self.normalize_before:
            x = self.after_norm(x)
        if self.output_layer is not None:
            x = self.output_layer(x)
        return x, tgt_mask

    def forward_one_step(self, tgt, tgt_mask, memory, memory_mask, cache=None):
        """Forward one step."""
        x = self.embed(tgt)
        if cache is None:
            cache = self.init_state()
        new_cache = []
        for ch, decoder in zip(cache, self.decoders):
            x, tgt_mask, memory, memory_mask = decoder(
                x, tgt_mask, memory, memory_mask, cache=None
            )
            new_cache.append(x)
        if self.normalize_before:
            y = self.after_norm(x[:, -1:, :])

        else:
            y = x[:, -1:, :]
        if self.output_layer is not None:
            y = self.output_layer(y)
            y = F.logsoftmax(y, axis=-1)
        return y, new_cache

    def init_state(self, x=None):
        """Get an initial state for decoding."""
        return [None for i in range(len(self.decoders))]
