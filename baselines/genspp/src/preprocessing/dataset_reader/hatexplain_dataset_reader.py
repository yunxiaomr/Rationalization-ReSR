import pandas as pd
import numpy as np
from statistics import mode

from src.preprocessing.dataset_reader.dataset_reader_interface import (
    DatasetReaderInterface,
)
from src.preprocessing.dataset_reader.dataset import Dataset
from src.preprocessing.embedding.glove_embedder import GloveEmbedder


class HatexplainDatasetReader(DatasetReaderInterface):
    def __init__(self, embedder_path: str, embedding_dim: int, max_filter: int | None = None):

        self.embedder_path = embedder_path
        self.embedding_dim = embedding_dim
        self.max_filter = max_filter

    def read(self, dataset_paths: list[str]) -> Dataset:

        dataframe_train: pd.DataFrame = self.__filter(pd.read_pickle(dataset_paths[0]))
        dataframe_test: pd.DataFrame = self.__filter(pd.read_pickle(dataset_paths[1]))
        dataframe_val: pd.DataFrame = self.__filter(pd.read_pickle(dataset_paths[2]))

        train_texts = self.__extract_texts(dataframe_train)
        test_texts = self.__extract_texts(dataframe_test)
        val_texts = self.__extract_texts(dataframe_val)
        train_texts = train_texts + val_texts

        train_labels = self.__extract_labels(dataframe_train)
        test_labels = self.__extract_labels(dataframe_test)
        val_labels = self.__extract_labels(dataframe_val)
        train_labels = train_labels + val_labels

        train_masks = self.__extract_masks(dataframe_train)
        test_masks = self.__extract_masks(dataframe_test)
        val_masks = self.__extract_masks(dataframe_val)
        train_masks = train_masks + val_masks

        embedder = GloveEmbedder(self.embedder_path, self.embedding_dim)

        return Dataset(
            train_texts,
            test_texts,
            train_labels,
            test_labels,
            train_masks,
            test_masks,
            embedder,
        )

    def __extract_texts(self, df: pd.DataFrame) -> list[list[str]]:

        texts = df["post_tokens"].to_list()

        return texts

    def __extract_labels(self, df: pd.DataFrame) -> list[int]:

        labels: list[int] = []

        annotations = df["annotators"]

        for annotation in annotations:
            votes = annotation["label"]
            votes = [x if x != 2 else 0 for x in votes]
            majority = mode(votes)
            labels.append(majority)

        return labels

    def __extract_masks(self, df: pd.DataFrame) -> list[list[np.ndarray]]:

        masks: list[list[np.ndarray]] = []

        rationales = df["rationales"]
        _texts = df["post_tokens"]
        lengths = [len(text) for text in _texts]

        for rationale, length in zip(rationales, lengths):
            mask_group: list[np.ndarray] = []
            if rationale is not None and len(rationale) != 0:
                for annotator_mask in rationale:
                    if len(annotator_mask) == length:
                        mask_group.append(np.array(annotator_mask))
            masks.append(mask_group)

        return masks

    def __filter(self, df: pd.DataFrame) -> pd.DataFrame:

        if self.max_filter is None:
            return df

        return df[df["post_tokens"].map(len) <= self.max_filter]
