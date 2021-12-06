import megengine.module as M
import megengine.functional as F
import megengine as mge


class PositionwiseFeedForward(M.Module):
    """Positionwise feed forward layer.

    :param int idim: input dimenstion
    :param int hidden_units: number of hidden units
    :param float dropout_rate: dropout rate

    """

    def __init__(self, idim, hidden_units, dropout_rate):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = M.Linear(idim, hidden_units)
        self.w_2 = M.Linear(hidden_units, idim)
        self.dropout = M.dropout.Dropout(dropout_rate)
        self.relu = M.ReLU()

    def forward(self, x):
        """Forward funciton."""
        return self.w_2(self.dropout(self.relu(self.w_1(x))))
