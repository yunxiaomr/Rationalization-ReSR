import random
import numpy as np
from abc import ABC

from src.genetic.crossover.crossover_interface import CrossoverInterface
from src.genetic.individual import Individual


class OnePointCrossover(CrossoverInterface, ABC):
    @staticmethod
    def apply_crossover(
        parent_1: Individual, parent_2: Individual
    ) -> tuple[np.ndarray, np.ndarray]:

        chromosome_1 = parent_1.chromosome
        chromosome_2 = parent_2.chromosome

        crossover_point = random.randrange(0, len(chromosome_1))

        parent_1_half_1 = chromosome_1[0:crossover_point]
        parent_1_half_2 = chromosome_1[crossover_point:]

        parent_2_half_1 = chromosome_2[0:crossover_point]
        parent_2_half_2 = chromosome_2[crossover_point:]

        child_1 = np.concatenate([parent_1_half_1, parent_2_half_2])
        child_2 = np.concatenate([parent_2_half_1, parent_1_half_2])

        return child_1, child_2
