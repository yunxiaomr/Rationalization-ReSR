from abc import ABC, abstractmethod
import numpy as np


class MutationInterface(ABC):
    @abstractmethod
    def mutate(self, chromosome: np.ndarray, mutation_prob: float) -> np.ndarray:
        pass
