import os
import hashlib
import getpass
import numpy as np


def abs_path(base_dir, path):
    """convert relative path to base_dir to absolute path
    :return: absolute path"""
    return os.path.join(base_dir, path)


def make_servable_name(exp_name, dataset_name, dep_files):
    """make a unique servable name.

    .. note::
        The resulting servable name is composed by the content of
        dependency files and the original dataset_name given.

    :param dataset_name: an dataset identifier, usually the argument
        passed to dataset.py:get
    :type dataset_name: str

    :param dep_files: files that the constrution of the dataset depends on.
    :type dep_files: list of str

    """

    def _md5(s):
        m = hashlib.md5()
        m.update(s)
        return m.hexdigest()

    parts = []
    for path in dep_files:
        with open(path, "rb") as f:
            parts.append(_md5(f.read()))
    return (
        exp_name + ":" + getpass.getuser() + ":" + ".".join(parts) + "." + dataset_name
    )


def _pad_data(x, length):
    _pad = 0
    return np.pad(x, (0, length - x.shape[0]), mode="constant", constant_values=_pad)


def _prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_data(x, max_len) for x in inputs])


def _pad_mel(inputs):
    _pad = 0

    def _pad_one(x, max_len):
        mel_len = x.shape[0]
        return np.pad(
            x, [[0, max_len - mel_len], [0, 0]], mode="constant", constant_values=_pad
        )

    max_len = max((x.shape[0] for x in inputs))
    return np.stack([_pad_one(x, max_len) for x in inputs])
