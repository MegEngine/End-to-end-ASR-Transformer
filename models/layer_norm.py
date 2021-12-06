import megengine.module as M
import megengine as mge
import megengine.functional as F
from megengine import Parameter
import numpy as np
from megengine.module import Dropout, Embedding, Linear, Module, Sequential


class AsrLayerNorm(Module):
    """Construct a layernorm module in the TF style (epsilon inside the square root)."""

    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.weight = Parameter(np.ones(hidden_size).astype(np.float32))
        self.bias = Parameter(np.zeros(hidden_size).astype(np.float32))
        self.variance_epsilon = eps

    def forward(self, x):
        u = F.mean(x, len(x.shape) - 1, True)
        s = F.mean((x - u) ** 2, len(x.shape) - 1, True)
        x = (x - u) / ((s + self.variance_epsilon) ** 0.5)
        return self.weight * x + self.bias


class LayerNorm(M.Module):
    def __init__(self, num_hidden):
        super(LayerNorm, self).__init__()
        self.layer_norm = AsrLayerNorm(num_hidden)

    def forward(self, x):
        x = self.layer_norm(x)
        return x
