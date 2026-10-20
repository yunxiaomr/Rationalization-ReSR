from abc import ABC, abstractmethod
import numpy as np

from src.genetic.individual import Individual


class CrossoverInterface(ABC):
    @staticmethod
    @abstractmethod
    def apply_crossover(
        parent_1: Individual, parent_2: Individual
    ) -> tuple[np.ndarray, np.ndarray]:
        pass
