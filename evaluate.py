import os
import sys
import time
from collections import OrderedDict
from time import strftime, gmtime
from dataset import AsrDataset, DataLoader, AsrCollator
from megengine.data import SequentialSampler, RandomSampler, DataLoader
from models.transformer import Model
import hparams as hp
import argparse
from tqdm import tqdm
import difflib
import megengine as mge
import megengine.functional as F


class Session:
    def __init__(self, args):
        with open(os.path.join(hp.dataset_root, "vocab.txt")) as f:
            self.vocab = [w.strip() for w in f.readlines()]
            self.vocab = ["<pad>"] + self.vocab
            print(f"Vocab Size: {len(self.vocab)}")
        self.pad_id = 0
        self.model = Model(hp.num_mels, len(self.vocab))  # .eval()
        ckpt = mge.load(args.model_path)
        self.model.load_state_dict(ckpt["model"], strict=False)
        self.model.eval()

        self.numerator = 0
        self.denominator = 0
        self.error_number = 0

    def reset_evaluate(self):
        self.numerator = 0
        self.denominator = 0
        self.error_number = 0

    def get_evaluate_acc(self):
        return (
            self.numerator / self.denominator,
            self.error_number / self.denominator,
            self.numerator,
            self.error_number,
            self.denominator,
        )

    def GetEditDistance(self, str1, str2):
        leven_cost = 0
        s = difflib.SequenceMatcher(None, str1, str2)
        for tag, i1, i2, j1, j2 in s.get_opcodes():
            # print('{:7} a[{}: {}] --> b[{}: {}] {} --> {}'.format(tag, i1, i2, j1, j2, str1[i1: i2], str2[j1: j2]))
            if tag == "replace":
                leven_cost += max(i2 - i1, j2 - j1)
            elif tag == "insert":
                leven_cost += j2 - j1
            elif tag == "delete":
                leven_cost += i2 - i1
        return leven_cost

    def evaluate(self, data):
        text_input, text_output, mel, pos_text, pos_mel, text_length, mel_length = data
        ys = self.model.forward(mel, mel_length, text_input, text_length, evaluate=True)
        mask = text_output != self.pad_id
        ys_str = ""
        text_output_str = ""
        for idex in ys[mask]:
            if self.vocab[idex] == "<eos>":
                ys_str += "."
            else:
                ys_str += self.vocab[idex]
        for idex in text_output[mask]:
            if self.vocab[idex] == "<eos>":
                text_output_str += "."
            else:
                text_output_str += self.vocab[idex]
        numerator = F.sum(ys[mask] == text_output[mask])
        denominator = F.sum(mask)
        edit_distance = self.GetEditDistance(ys_str, text_output_str)
        if edit_distance <= denominator.item() - numerator.item():
            self.error_number += edit_distance
        else:
            self.error_number += denominator.item() - numerator.item()
        self.numerator += numerator.item()
        self.denominator += denominator.item()


def main():
    os.makedirs(hp.checkpoint_path, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", default="dev", choices=["dev", "test", "train"])
    args = parser.parse_args()

    dataset = AsrDataset(args.dataset)
    sess = Session(args)

    val_sampler = SequentialSampler(dataset=dataset, batch_size=32)

    dataloader = DataLoader(
        dataset=dataset, sampler=val_sampler, collator=AsrCollator()
    )
    sess.reset_evaluate()
    for idx, data in enumerate(tqdm(dataloader)):
        text_input, text_output, mel, pos_text, pos_mel, text_length, mel_length = data
        sess.evaluate(data)
    acc, wer, numerator, error_number, denominator = sess.get_evaluate_acc()
    print("ACC: ", acc, "CER", wer)


if __name__ == "__main__":
    main()
