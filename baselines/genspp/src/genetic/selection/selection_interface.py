from abc import ABC, abstractmethod

from src.genetic.individual import Individual


class SelectionInterface(ABC):
    @staticmethod
    @abstractmethod
    def select(
        population: list[Individual], selection_rate: float, workers=1
    ) -> list[tuple[Individual, Individual]]:
        pass
