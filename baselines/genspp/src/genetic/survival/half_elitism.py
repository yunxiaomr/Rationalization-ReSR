import numpy as np

from abc import ABC
from multiprocessing.pool import ThreadPool

from src.genetic.survival.survival_interface import SurvivalInterface
from src.genetic.individual import Individual
from src.genetic.fitness_utils import normalized_fitness_weights, safe_fitness


class HalfElitismSurvival(SurvivalInterface, ABC):
    @staticmethod
    def survival_select(
        population: list[Individual], population_size: int, workers=1
    ) -> list[Individual]:

        if workers == 1:
            for individual in population:
                _ = individual.fitness

        else:
            with ThreadPool(processes=workers) as pool:
                pool.map(lambda x: x.fitness, population)

        population.sort(reverse=True, key=lambda x: safe_fitness(x.fitness))

        half = int(population_size / 2)
        new_population = population[0:half]

        selection_group = population[half:]
        fitness_values = [individual.fitness for individual in selection_group]
        probabilities = normalized_fitness_weights(fitness_values, min_positive=half)

        winners = np.random.choice(np.array(selection_group), half, replace=False, p=probabilities)
        winners = winners.tolist()

        losers = [individual for individual in selection_group if individual not in winners]

        new_population.extend(winners)
        new_population.extend(losers)

        return new_population
