import numpy as np
from tqdm import tqdm

from src.preprocessing.embedding.embedder_interface import (
    EmbedderInterface,
    EmbeddingInputTypes,
)


class Dataset:
    def __init__(
        self,
        texts_train: list[list[str]],
        texts_test: list[list[str]],
        labels_train: list[str | int],
        labels_test: list[str | int],
        highlight_masks_train: list[list[np.ndarray]],
        highlight_masks_test: list[list[np.ndarray]],
        embedder: EmbedderInterface,
        texts_val: list[list[str]] | None = None,
        labels_val: list[str | int] | None = None,
        highlight_masks_val: list[list[np.ndarray]] | None = None,
    ):

        assert len(texts_train) == len(labels_train) == len(highlight_masks_train)
        assert len(texts_test) == len(labels_test) == len(highlight_masks_test)
        texts_val = [] if texts_val is None else texts_val
        labels_val = [] if labels_val is None else labels_val
        highlight_masks_val = [] if highlight_masks_val is None else highlight_masks_val
        assert len(texts_val) == len(labels_val) == len(highlight_masks_val)

        self.n_samples_train = len(texts_train)
        self.n_samples_val = len(texts_val)
        self.n_samples_test = len(texts_test)
        self.texts = texts_train + texts_val + texts_test
        self.labels = labels_train + labels_val + labels_test
        self.highlight_masks = highlight_masks_train + highlight_masks_val + highlight_masks_test
        self.embedder = embedder

        self.tokenizer, self.detokenizer, self.max_len = self.__build_tokenizer()
        self.vocab_size = len(self.tokenizer) + 1

        tokenized_texts, self.masks = self.__tokenize_and_pad()
        self.embedded_texts = self.__embed_texts(tokenized_texts)

        self.labels_encoder = self.__build_labels_encoder()
        self.encoded_labels = self.__encode_labels()

    @property
    def embedding_dim(self):
        return self.embedder.embedding_dim

    def get_train_and_test_sets(self) -> tuple[tuple, tuple]:

        print("{} samples in the training set".format(self.n_samples_train))
        print("{} samples in the test set".format(self.n_samples_test))

        texts_train = self.embedded_texts[0 : self.n_samples_train]
        masks_train = self.masks[0 : self.n_samples_train]
        labels_train = self.encoded_labels[0 : self.n_samples_train]
        highlight_masks_train = self.highlight_masks[0 : self.n_samples_train]

        test_start = self.n_samples_train + self.n_samples_val
        texts_test = self.embedded_texts[test_start:]
        masks_test = self.masks[test_start:]
        labels_test = self.encoded_labels[test_start:]
        highlight_masks_test = self.highlight_masks[test_start:]

        x_train = [texts_train, masks_train]
        x_test = [texts_test, masks_test]

        train_set = (x_train, labels_train, highlight_masks_train)
        test_set = (x_test, labels_test, highlight_masks_test)

        return train_set, test_set

    def get_train_val_test_sets(self):
        if self.n_samples_val <= 0:
            train_set, test_set = self.get_train_and_test_sets()
            return train_set, None, test_set

        train_end = self.n_samples_train
        val_end = train_end + self.n_samples_val
        train_set = (
            [self.embedded_texts[:train_end], self.masks[:train_end]],
            self.encoded_labels[:train_end],
            self.highlight_masks[:train_end],
        )
        val_set = (
            [self.embedded_texts[train_end:val_end], self.masks[train_end:val_end]],
            self.encoded_labels[train_end:val_end],
            self.highlight_masks[train_end:val_end],
        )
        test_set = (
            [self.embedded_texts[val_end:], self.masks[val_end:]],
            self.encoded_labels[val_end:],
            self.highlight_masks[val_end:],
        )
        return train_set, val_set, test_set

    def get_train_and_test_texts(self) -> tuple[list[list[str]], list[list[str]]]:

        texts_train = self.texts[0 : self.n_samples_train]
        test_start = self.n_samples_train + self.n_samples_val
        texts_test = self.texts[test_start:]

        return texts_train, texts_test

    def get_classes_count(self) -> int:
        return len(self.labels_encoder)

    def __build_tokenizer(self) -> tuple[dict[str, int], dict[int, str], int]:

        tokenizer: dict[str, int] = dict()
        detokenizer: dict[int, str] = dict()
        max_len = 0

        token_number = 1

        print("Building tokenizer...")

        for text in tqdm(self.texts[0 : self.n_samples_train]):
            if len(text) > max_len:
                max_len = len(text)
            for word in text:
                if word not in tokenizer:
                    tokenizer[word] = token_number
                    detokenizer[token_number] = word
                    token_number += 1

        print("Max length: {}".format(max_len))

        return tokenizer, detokenizer, max_len

    def __tokenize_and_pad(self) -> tuple[list[list[int]], np.ndarray]:

        tokenized_texts: list[list[int]] = []
        masks: list[np.ndarray] = []

        print("Tokenizing texts...")

        for text in tqdm(self.texts):
            tokenized_sequence: list[int] = []

            for word in text:
                if word in self.tokenizer:
                    token = self.tokenizer[word]
                else:
                    token = self.vocab_size

                tokenized_sequence.append(token)

            mask = [1] * len(text)

            pad_len = self.max_len - len(text)
            if pad_len > 0:
                pad_sequence = [0] * pad_len
                tokenized_sequence.extend(pad_sequence)
                mask.extend(pad_sequence)
            elif pad_len < 0:
                tokenized_sequence = tokenized_sequence[0:pad_len]
                mask = mask[0:pad_len]

            tokenized_texts.append(tokenized_sequence)
            masks.append(np.array(mask))

        return tokenized_texts, np.array(masks)

    def __embed_texts(self, tokenized_texts: list[list[int]]) -> np.ndarray:

        print("Embedding texts...")

        if not tokenized_texts:
            return np.empty((0, 0, self.embedding_dim), dtype=np.float32)
        embedded_texts = np.empty(
            (len(tokenized_texts), len(tokenized_texts[0]), self.embedding_dim),
            dtype=np.float32,
        )
        for text_index, text in enumerate(tqdm(tokenized_texts)):
            for token_index, token in enumerate(text):
                embedding_type = self.embedder.get_input_type()
                if embedding_type == EmbeddingInputTypes.TOKEN:
                    embedding = self.embedder.embed(token)
                elif embedding_type == EmbeddingInputTypes.TEXT:
                    word = self.detokenizer.get(token, None)
                    embedding = self.embedder.embed(word)
                else:
                    raise Exception("Invalid EmbeddingInputType: {}".format(embedding_type))
                embedded_texts[text_index, token_index] = np.asarray(embedding, dtype=np.float32)

        return embedded_texts

    def __build_labels_encoder(self) -> dict[str | int, np.ndarray]:

        label_encoder: dict[str | int, np.ndarray] = dict()

        unique_labels: list[str | int] = sorted(
            set(self.labels),
            key=lambda value: (type(value).__name__, str(value)),
        )

        for i, label in enumerate(unique_labels):
            one_hot_encoding = [0] * len(unique_labels)
            one_hot_encoding[i] = 1
            label_encoder[label] = np.array(one_hot_encoding)

        return label_encoder

    def __encode_labels(self) -> np.ndarray:

        encoded_labels: list[np.ndarray] = []

        print("Encoding labels...")

        for label in tqdm(self.labels):
            encoded_label = self.labels_encoder[label]
            encoded_labels.append(encoded_label)

        return np.array(encoded_labels)
