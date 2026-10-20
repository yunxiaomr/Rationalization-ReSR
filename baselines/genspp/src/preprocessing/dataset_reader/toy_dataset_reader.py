import pandas as pd
import numpy as np

from src.preprocessing.dataset_reader.dataset_reader_interface import (
    DatasetReaderInterface,
)
from src.preprocessing.dataset_reader.dataset import Dataset
from src.preprocessing.embedding.one_hot_embedder import OneHotEmbedder


class ToyDatasetReader(DatasetReaderInterface):
    def __init__(self, max_len: int, train_percentage: float):

        assert 0.0 < train_percentage < 1.0

        self.max_len = max_len
        self.train_percentage = train_percentage

    def read(self, dataset_paths: list[str]) -> Dataset:

        dataset_path = dataset_paths[0]
        dataframe: pd.DataFrame = pd.read_pickle(dataset_path)

        full_texts: list[str] = dataframe["text"].to_list()
        labels: list[str] = dataframe["label"].to_list()

        highlight_indices: list[list[int]] = dataframe["structure_indexes"].to_list()
        highlight_masks = self.__build_highlight_masks(highlight_indices)

        texts: list[list[str]] = []
        for text in full_texts:
            split_text: list[str] = [*text]
            texts.append(split_text)

        embedder = OneHotEmbedder(vocab_size=26)

        train_count = int(len(texts) * self.train_percentage)

        texts_train = texts[0:train_count]
        texts_test = texts[train_count:]

        labels_train = labels[0:train_count]
        labels_test = labels[train_count:]

        masks_train = highlight_masks[0:train_count]
        masks_test = highlight_masks[train_count:]

        return Dataset(
            texts_train,
            texts_test,
            labels_train,
            labels_test,
            masks_train,
            masks_test,
            embedder,
        )

    def __build_highlight_masks(self, highlight_indices: list[list[int]]) -> list[list[np.ndarray]]:

        highlight_masks: list[list[[np.ndarray]]] = []

        for indices in highlight_indices:
            mask = [0] * self.max_len
            for index in indices:
                mask[index] = 1
            highlight_masks.append([np.array(mask)])

        return highlight_masks
