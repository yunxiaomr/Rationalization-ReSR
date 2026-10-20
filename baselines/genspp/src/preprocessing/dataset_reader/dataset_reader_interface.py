from abc import ABC, abstractmethod

from src.preprocessing.dataset_reader.dataset import Dataset


class DatasetReaderInterface(ABC):
    @abstractmethod
    def read(self, dataset_paths: list[str]) -> Dataset:
        pass
