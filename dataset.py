import os
import numpy as np
import collections
import megengine.module as M
import megengine.functional as F
import megengine as mge
from megengine.data.dataset import Dataset
from megengine.data import DataLoader
import hparams as hp
from megengine.data import Collator


class AsrDataset(Dataset):
    def __init__(self, data_set="train"):
        """
        Args:
            root_dir (string): Directory with all the spectrograms.

        """
        self.metas = self.load_metas(hp.dataset_root, data_set)

    def load_metas(self, root, data_set):  # fix a bug
        metas = []
        with open(os.path.join(root, f"{data_set}.txt")) as f:
            for line in f.readlines():
                info = line.split("|")
                metas.append(
                    {
                        "mel_path": os.path.join(root, info[0]),
                        "frames": info[1],
                        "token_ids_str": info[2],
                        "speaker": info[3],
                    }
                )
        return metas

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        meta = self.metas[idx]
        token_ids = [int(i) for i in meta["token_ids_str"].split(" ")]
        text = np.array(token_ids, dtype=np.int32)
        mel = np.load(meta["mel_path"])
        text_input = text[:-1]
        text_output = text[1:]
        text_length = text_input.shape[0]
        pos_text = np.arange(1, text_length + 1)
        pos_mel = np.arange(1, mel.shape[0] + 1)
        return {
            "text": text,
            "text_input": text_input,
            "text_output": text_output,
            "text_length": text_length,
            "mel": mel,
            "pos_mel": pos_mel,
            "pos_text": pos_text,
        }


class AsrCollator(Collator):
    def __init__(self, pad_value: float = 0.0):
        super().__init__()
        self.pad_value = pad_value

    def apply(self, batch):
        # Puts each data field into a tensor with outer dimension batch size
        if isinstance(batch[0], collections.Mapping):
            text = [d["text"] for d in batch]
            text_input = [d["text_input"] for d in batch]
            text_output = [d["text_output"] for d in batch]
            text_length = [d["text_length"] for d in batch]
            mel = [d["mel"] for d in batch]
            mel_length = [d["mel"].shape[0] for d in batch]
            pos_mel = [d["pos_mel"] for d in batch]
            pos_text = [d["pos_text"] for d in batch]

            text = [
                i
                for i, _ in sorted(
                    zip(text, mel_length), key=lambda x: x[1], reverse=True
                )
            ]
            text_input = [
                i
                for i, _ in sorted(
                    zip(text_input, mel_length), key=lambda x: x[1], reverse=True
                )
            ]
            text_output = [
                i
                for i, _ in sorted(
                    zip(text_output, mel_length), key=lambda x: x[1], reverse=True
                )
            ]
            text_length = [
                i
                for i, _ in sorted(
                    zip(text_length, mel_length), key=lambda x: x[1], reverse=True
                )
            ]
            mel = [
                i
                for i, _ in sorted(
                    zip(mel, mel_length), key=lambda x: x[1], reverse=True
                )
            ]
            pos_text = [
                i
                for i, _ in sorted(
                    zip(pos_text, mel_length), key=lambda x: x[1], reverse=True
                )
            ]
            pos_mel = [
                i
                for i, _ in sorted(
                    zip(pos_mel, mel_length), key=lambda x: x[1], reverse=True
                )
            ]
            mel_length = sorted(mel_length, reverse=True)

            # PAD sequences with largest length of the batch
            text_input = _prepare_data(text_input).astype(np.int32)
            text_output = _prepare_data(text_output).astype(np.int32)
            mel = _pad_mel(mel)
            pos_mel = _prepare_data(pos_mel).astype(np.int32)
            pos_text = _prepare_data(pos_text).astype(np.int32)

            return (
                mge.Tensor(text_input),
                mge.Tensor(text_output),
                mge.Tensor(mel),
                mge.Tensor(pos_text),
                mge.Tensor(pos_mel),
                mge.Tensor(text_length),
                mge.Tensor(mel_length),
            )

        raise TypeError(
            (
                "batch must contain tensors, numbers, dicts or lists; found {}".format(
                    type(batch[0])
                )
            )
        )


def collate_fn_transformer_test(batch):
    # Puts each data field into a tensor with outer dimension batch size
    # if isinstance(batch[0], collections.Mapping):
    text = [batch["text"]]  # for d in batch]
    text_input = batch["text_input"]
    text_output = batch["text_output"]
    text_length = batch["text_length"]
    mel = [batch["mel"]]
    mel_length = [batch["mel"].shape[1]]
    pos_mel = batch["pos_mel"]
    pos_text = batch["pos_text"]
    text = [
        i for i, _ in sorted(zip(text, mel_length), key=lambda x: x[1], reverse=True)
    ]
    text_input = [
        i
        for i, _ in sorted(
            zip(text_input, mel_length), key=lambda x: x[1], reverse=True
        )
    ]
    text_output = [
        i
        for i, _ in sorted(
            zip(text_output, mel_length), key=lambda x: x[1], reverse=True
        )
    ]
    text_length = [
        i
        for i, _ in sorted(
            zip(text_length, mel_length), key=lambda x: x[1], reverse=True
        )
    ]
    mel = [i for i, _ in sorted(zip(mel, mel_length), key=lambda x: x[1], reverse=True)]
    pos_text = [
        i
        for i, _ in sorted(zip(pos_text, mel_length), key=lambda x: x[1], reverse=True)
    ]
    pos_mel = [
        i for i, _ in sorted(zip(pos_mel, mel_length), key=lambda x: x[1], reverse=True)
    ]
    mel_length = sorted(mel_length, reverse=True)

    # PAD sequences with largest length of the batch
    text_input = _prepare_data(text_input).astype(np.int32)
    text_output = _prepare_data(text_output).astype(np.int32)
    mel = _pad_mel(mel[0])
    pos_mel = _prepare_data(pos_mel).astype(np.int32)
    pos_text = _prepare_data(pos_text).astype(np.int32)

    return (
        mge.Tensor(text_input),
        mge.Tensor(text_output),
        mge.Tensor(mel),
        mge.Tensor(pos_text),
        mge.Tensor(pos_mel),
        mge.Tensor(text_length),
        mge.Tensor(mel_length),
    )

    raise TypeError(
        (
            "batch must contain tensors, numbers, dicts or lists; found {}".format(
                type(batch[0])
            )
        )
    )


############################ Utils ###################################


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
