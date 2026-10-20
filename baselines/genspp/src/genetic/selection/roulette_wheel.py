import random
from multiprocessing.pool import ThreadPool
from abc import ABC

from src.genetic.selection.selection_interface import SelectionInterface
from src.genetic.individual import Individual
from src.genetic.fitness_utils import normalized_fitness_weights


class RouletteWheelSelection(SelectionInterface, ABC):
    @staticmethod
    def select(
        population: list[Individual], selection_rate: float, workers=1
    ) -> list[tuple[Individual, Individual]]:

        with ThreadPool(processes=workers) as pool:
            pool.map(lambda x: x.fitness, population)

        fitness_values = [individual.fitness for individual in population]
        probabilities = normalized_fitness_weights(fitness_values)

        n_couples = int(selection_rate * len(population))
        couples: list[tuple[Individual, Individual]] = []

        for _ in range(n_couples):
            couple = random.choices(population, weights=probabilities, k=2)
            couples.append(tuple(couple))

        return couples
