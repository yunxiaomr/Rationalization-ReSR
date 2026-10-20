import numpy as np

from src.preprocessing.embedding.embedder_interface import (
    EmbedderInterface,
    EmbeddingInputTypes,
)


class OneHotEmbedder(EmbedderInterface):
    def __init__(self, vocab_size: int):

        super().__init__(vocab_size)

        self.vocab_size = vocab_size

        self.embedding_dict: dict[int, np.ndarray] = dict()
        for i in range(vocab_size):
            encoding = [0] * self.vocab_size
            encoding[i] = 1
            self.embedding_dict[i + 1] = np.array(encoding)

    def get_input_type(self) -> int:
        return EmbeddingInputTypes.TOKEN

    def embed(self, token: int | str) -> np.ndarray:

        if token not in self.embedding_dict:
            return np.zeros(self.vocab_size)

        return self.embedding_dict[token]
