from abc import ABC, abstractmethod

import numpy as np


class EmbeddingInputTypes:
    TOKEN = 0
    TEXT = 1


class EmbedderInterface(ABC):
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim

    @abstractmethod
    def get_input_type(self) -> int:
        pass

    @abstractmethod
    def embed(self, token: int | str) -> np.ndarray:
        pass
