from abc import ABC, abstractmethod

from src.genetic.individual import Individual


class SurvivalInterface(ABC):
    @staticmethod
    @abstractmethod
    def survival_select(
        population: list[Individual], population_size: int, workers=1
    ) -> list[Individual]:
        pass
