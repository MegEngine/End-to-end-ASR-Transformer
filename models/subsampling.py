import megengine.module as M
import megengine.functional as F
import megengine as mge
from .embedding import PositionalEncoding


class Conv2dSubsampling(M.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate

    """

    def __init__(self, idim, odim, dropout_rate):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling, self).__init__()
        self.odim = odim
        self.idim = idim
        self.linear = M.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim)
        self.PositionalEncoding = PositionalEncoding(odim, dropout_rate)
        self.conv = M.Sequential(
            M.Conv2d(1, odim, 3, 2), M.ReLU(), M.Conv2d(odim, odim, 3, 2), M.ReLU()
        )
        self.out = M.Sequential(
            M.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
            PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x."""
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        x = self.conv(x)
        b, c, t, f = x.shape

        x = self.out(F.transpose(x, (0, 2, 1, 3)).reshape(b, t, c * f))

        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]
