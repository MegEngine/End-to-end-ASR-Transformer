import os
import numpy as np
import hparams as hp
from tqdm import tqdm
from functools import partial
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from utils import audio


def build_from_path(in_dir, out_dir, num_workers=1, index=1):
    """Preprocesses the Biaobei dataset from a given input path into a given output directory.
    Args:
        in_dir: The directory where you have downloaded the Biaobei dataset
        out_dir: The directory to write the output into
        num_workers: Optional number of worker processes to parallelize across
    Returns:
        A list of tuples describing the training examples. This should be written to train.txt
    """
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    speakers = [s for s in os.listdir(in_dir) if s.startswith("G")]
    print("speaker num: ", len(speakers))
    for speaker in tqdm(speakers):
        sample_ids = [
            s[:-4]
            for s in os.listdir(os.path.join(in_dir, speaker))
            if s.endswith("txt")
        ]
        # print(f"{speaker}: {len(sample_ids)}")
        for sample_id in sample_ids:
            text_path = os.path.join(in_dir, speaker, f"{sample_id}.txt")
            wav_path = os.path.join(in_dir, speaker, f"{sample_id}.wav")
            futures.append(
                executor.submit(
                    partial(
                        _process_utterance, out_dir, index, wav_path, text_path, speaker
                    )
                )
            )
            index += 1
    return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, index, wav_path, text_path, speaker):
    """Preprocesses a single utterance audio/text pair.
    This writes the mel spectrograms to disk and returns a tuple to write
    to the train.txt file.
    Args:
        out_dir: The directory to write the spectrograms into
        index: The numeric index to use in the spectrogram filenames.
        wav_path: Path to the audio file containing the speech input
        text: The text spoken in the input audio file

    Returns:
        A (mel_filename, n_frames, text) tuple to write to train.txt
    """
    with open(text_path) as f:
        text = f.readline().strip()
    # Load autio into a numpy narray
    wav = audio.load_wav(wav_path)

    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
    n_frames = mel_spectrogram.shape[1]

    # Write the spectrograms to disk:
    mel_filename = "asr-mel-%07d.npy" % index
    np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

    # Return a tuple describing this training example:
    return (mel_filename, n_frames, text, speaker)


def write_metadata(metadata, out_dir, data_set):
    with open(os.path.join(out_dir, f"{data_set}.txt"), "w", encoding="utf-8") as f:
        for m in metadata:
            f.write("|".join([str(x) for x in m]) + "\n")
    frames = sum([m[1] for m in metadata])
    hours = frames * hp.frame_shift_ms / (3600 * 1000)
    print(
        "Wrote %d utterances, %d frames (%.2f hours)" % (len(metadata), frames, hours)
    )


def get_dict(texts, out_dir):
    vocab = set()
    # vocab.add("<unk>")
    # vocab.add("<sos>")
    # vocab.add("<eos>")
    for text in texts:
        for ch in text:
            vocab.add(ch)
    vocab = list(vocab)
    vocab.sort()
    vocab = ["<unk>", "<sos>", "<eos>"] + vocab
    with open(os.path.join(out_dir, "vocab.txt"), "w+") as f:
        f.writelines([w + "\n" for w in vocab])
    word2num = {}
    for w in vocab:
        word2num[w] = len(word2num) + 1
    print("Vocab size: ", len(vocab))
    return word2num


def word_to_num(metadata, word2num):
    data = []
    for (mel_filename, n_frames, text, speaker) in metadata:
        tokenids = (
            [word2num["<sos>"]]
            + [word2num[w] if w in word2num else word2num["<unk>"] for w in text]
            + [word2num["<eos>"]]
        )
        tokenids = " ".join([str(t) for t in tokenids])
        data.append((mel_filename, n_frames, tokenids, speaker))
    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        required=True,
        help="The directory where you have downloaded the LJ Speech dataset",
    )
    parser.add_argument(
        "--out-dir", required=True, help="The directory to write the output into"
    )
    parser.add_argument("--num-workers", type=int, default=cpu_count())
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_metadata = build_from_path(
        os.path.join(args.data_dir, "train"), args.out_dir, args.num_workers
    )
    word2num = get_dict([m[2] for m in train_metadata], args.out_dir)
    train_metadata = word_to_num(train_metadata, word2num)
    write_metadata(train_metadata, args.out_dir, "train")

    dev_metadata = build_from_path(
        os.path.join(args.data_dir, "dev"),
        args.out_dir,
        args.num_workers,
        len(train_metadata) + 1,
    )
    dev_metadata = word_to_num(dev_metadata, word2num)
    write_metadata(dev_metadata, args.out_dir, "dev")

    text_metadata = build_from_path(
        os.path.join(args.data_dir, "test"),
        args.out_dir,
        args.num_workers,
        len(train_metadata) + len(dev_metadata) + 1,
    )
    text_metadata = word_to_num(text_metadata, word2num)
    write_metadata(text_metadata, args.out_dir, "test")
