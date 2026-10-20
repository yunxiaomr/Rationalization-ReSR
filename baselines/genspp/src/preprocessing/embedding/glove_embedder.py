import numpy as np

from src.preprocessing.embedding.embedder_interface import (
    EmbedderInterface,
    EmbeddingInputTypes,
)


class GloveEmbedder(EmbedderInterface):
    def __init__(self, path: str, embedding_dim: int):

        assert embedding_dim in [25, 50, 100, 200]

        super().__init__(embedding_dim)

        self.path = path

        self.glove = self.__load_glove()

        rng = np.random.RandomState(100)
        self.unk_placeholder = rng.uniform(low=-0.05, high=0.05, size=embedding_dim).astype(
            np.float32
        )

    def __load_glove(self) -> dict[str, np.ndarray]:

        print("Loading GloVe Model...")

        vocabulary: dict[str, np.ndarray] = dict()
        with open(self.path.format(self.embedding_dim), encoding="utf8") as handle:
            for line in handle:
                splits = line.split()
                vocabulary[splits[0]] = np.array([float(val) for val in splits[1:]])

        print("GloVe model loaded!")

        return vocabulary

    def get_input_type(self) -> int:
        return EmbeddingInputTypes.TEXT

    def embed(self, token: int | str) -> np.ndarray:

        if token is None:
            return np.zeros(self.embedding_dim)

        return self.glove.get(token, self.unk_placeholder)
