from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchtext import vocab
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm


class HighlightDataset(Dataset):
    def __init__(self, texts, masks, labels, highlights, sample_ids):
        self.texts = texts
        self.masks = masks
        self.labels = labels
        self.highlights = highlights
        self.sample_ids = sample_ids

    def __getitem__(self, index):
        return (
            self.texts[index],
            self.masks[index],
            self.labels[index],
            self.highlights[index],
            self.sample_ids[index],
        )

    def __len__(self):
        return len(self.texts)


class OneHotEmbedderCollator:
    def __init__(
        self,
    ):
        self.vocabulary = None
        self.vocab_size = None
        self.embedding_matrix = None

    def fit(self, df: pd.DataFrame):
        tokens = []
        with tqdm(desc="Fitting text sequences", leave=True, position=0, total=df.shape[0]) as pbar:
            for seq_id, (text, highlight) in enumerate(zip(df.text.values, df.highlight.values)):
                text_tokens = list(text)
                tokens.append(text_tokens)

                pbar.update(1)

        self.vocabulary = build_vocab_from_iterator(
            iterator=tokens, specials=["<PAD>"], special_first=True
        )
        self.vocab_size = len(self.vocabulary)

        self.embedding_matrix = th.nn.functional.one_hot(
            th.arange(0, self.vocab_size), num_classes=self.vocab_size
        )
        self.embedding_matrix[0] *= 0
        self.embedding_matrix = self.embedding_matrix.to(th.float32)

    def embed(self, text):
        return [[self.vocabulary[token] for token in seq] for seq in list(text)]

    def __call__(self, batch):
        texts, masks, labels, highlights, sample_ids = zip(*batch)

        input_ids = [th.tensor(self.embed(text), dtype=th.int32).ravel() for text in texts]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)

        th_masks = [th.tensor(mask) for mask in masks]
        th_masks = pad_sequence(th_masks, batch_first=True, padding_value=0)

        th_labels = th.tensor(labels)

        th_highlights = pad_sequence(
            [th.tensor(highlight, dtype=th.int32) for highlight in highlights],
            batch_first=True,
            padding_value=-1,
        )

        th_sample_ids = th.tensor(sample_ids)

        return input_ids, th_masks, th_sample_ids, th_labels, th_highlights


class GloVeEmbedderCollator:
    def __init__(self, name="twitter.27B", dim=25, use_pretrained_only=False, cache_dir=None):
        cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else Path(__file__).resolve().parents[1] / "embeddings"
        )
        local_vectors = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "glove"
            / "twitter"
            / f"glove.{name}.{dim}d.txt"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_vectors = cache_dir / local_vectors.name
        if name == "twitter.27B" and not any(
            path.is_file()
            for path in (local_vectors, cached_vectors, cache_dir / f"{local_vectors.name}.pt")
        ):
            raise FileNotFoundError(
                f"Twitter GloVe {dim}d vectors are missing. Extract {local_vectors.name} "
                f"to {local_vectors.parent}. See README.md#data for the download link."
            )
        if name == "twitter.27B" and local_vectors.is_file():
            if cached_vectors.is_symlink() and not cached_vectors.exists():
                cached_vectors.unlink()
            if not cached_vectors.exists():
                cached_vectors.symlink_to(local_vectors)
        self.embedding_model = vocab.GloVe(name=name, dim=dim, cache=str(cache_dir))
        self.vocabulary = None
        self.vocab_size = None
        self.embedding_matrix = None
        self.use_pretrained_only = use_pretrained_only

    def fit(self, df: pd.DataFrame):
        if self.use_pretrained_only:
            self.vocabulary = vocab.vocab(
                self.embedding_model.stoi,
                min_freq=0,
                specials=["<PAD>"],
                special_first=True,
            )
        else:
            input_texts = [
                text.split(" ") if type(text) == str else text.tolist() for text in df.text.values
            ]
            self.vocabulary = build_vocab_from_iterator(
                iterator=iter(input_texts), specials=["<PAD>"], special_first=True
            )

        self.vocabulary.set_default_index(0)
        self.vocab_size = len(self.vocabulary)
        self.embedding_matrix = self.embedding_model.get_vecs_by_tokens(self.vocabulary.get_itos())

    def __call__(self, batch):
        texts, masks, labels, highlights, sample_ids = zip(*batch)

        input_ids = []
        for text in texts:
            if type(text) == str:
                input_ids.append(th.tensor(self.vocabulary(text.split(" "))))
            elif type(text) == np.ndarray:
                input_ids.append(th.tensor(self.vocabulary(text.tolist())))
            else:
                input_ids.append(th.tensor(self.vocabulary(text)))

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)

        th_masks = [th.tensor(mask, dtype=th.float32) for mask in masks]
        th_masks = pad_sequence(th_masks, batch_first=True, padding_value=0)

        th_labels = th.tensor(labels)

        th_highlights = pad_sequence(
            [th.tensor(highlight, dtype=th.int32) for highlight in highlights],
            batch_first=True,
            padding_value=-1,
        )

        th_sample_ids = th.tensor(sample_ids, dtype=th.int32)

        return input_ids, th_masks, th_sample_ids, th_labels, th_highlights


class HighlightExtractor:
    def __init__(self, text_splitter=lambda t: t.split(" "), text_merger=lambda t: " ".join(t)):
        self.text_splitter = text_splitter
        self.text_merger = text_merger

    def to_readable_format(self, df):
        content = []
        for _, row in df.iterrows():
            content_row = f"""
               Text: {row["text"]}
            Hl True: {row["hl_true"]}
            Hl Pred: {row["hl_hat"]}
            Y True:  {row["y_true"]}
            Y Pred:  {row["y_hat"]}
            """
            content.append(content_row)

        return "\n\n".join(content)

    def __call__(self, predictions, vocabulary, save_path: Path, seed: int):
        prediction_dict = {}

        for batch_predictions in predictions:
            y_true = batch_predictions["y_true"].numpy()
            y_hat = batch_predictions["y_hat"].numpy()
            hl_true = batch_predictions["hl_true"].numpy()
            hl_hat = batch_predictions["hl_hat"].numpy()
            input_ids = batch_predictions["input_ids"].numpy()
            sample_ids = batch_predictions["sample_ids"].numpy()

            for s_y_true, s_y_hat, s_hl_true, s_hl_hat, s_input_ids, s_id in zip(
                y_true, y_hat, hl_true, hl_hat, input_ids, sample_ids
            ):
                valid_indexes = np.where(s_input_ids != 0)[0]
                s_text = self.text_merger(vocabulary.lookup_tokens(s_input_ids[valid_indexes]))

                s_hl_true_text = self.text_merger(
                    [
                        token if hl_mask else "_"
                        for token, hl_mask in zip(
                            self.text_splitter(s_text), s_hl_true[valid_indexes]
                        )
                    ]
                )

                s_hl_hat_text = self.text_merger(
                    [
                        token if hl_mask else "_"
                        for token, hl_mask in zip(
                            self.text_splitter(s_text), s_hl_hat[valid_indexes]
                        )
                    ]
                )

                prediction_dict.setdefault("text", []).append(s_text)
                prediction_dict.setdefault("hl_true", []).append(s_hl_true_text)
                prediction_dict.setdefault("hl_hat", []).append(s_hl_hat_text)
                prediction_dict.setdefault("y_true", []).append(s_y_true)
                prediction_dict.setdefault("y_hat", []).append(s_y_hat)
                prediction_dict.setdefault("sample_id", []).append(s_id)

        prediction_df = pd.DataFrame.from_dict(prediction_dict)

        readable_format = self.to_readable_format(prediction_df)

        prediction_df.to_csv(save_path.joinpath(f"predictions_seed={seed}.csv"), index=False)
        with save_path.joinpath(f"predictions_seed={seed}.txt").open("w") as f:
            f.write(readable_format)


def get_glove_embedding(glove_embedding_path: Path):
    with glove_embedding_path.open("rt", encoding="utf-8") as f:
        lines = f.readlines()
        embedding = []
        word2idx = {}
        for indx, line in enumerate(lines):
            word, emb = line.split()[0], line.split()[1:]
            vector = [float(x) for x in emb]
            if indx == 0:
                embedding.append(np.zeros(len(vector)))
            embedding.append(vector)
            word2idx[word] = indx + 1

        embedding = np.array(embedding, dtype=np.float32)

        return embedding, word2idx
