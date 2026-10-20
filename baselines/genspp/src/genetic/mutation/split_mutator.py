import numpy as np

from src.genetic.mutation.mutation_interface import MutationInterface


class SplitMutation(MutationInterface):
    def __init__(
        self,
        sections: list[tuple[int, int]],
        mutations: list[MutationInterface],
        probabilities: list[float],
    ):

        assert len(sections) == len(mutations) == len(probabilities)

        self.sections = sections
        self.mutations = mutations
        self.probabilities = probabilities

    def mutate(self, chromosome: np.ndarray, mutation_prob: float) -> np.ndarray:

        mutated_chromosome: list[float] = []

        for section, mutation, probability in zip(
            self.sections, self.mutations, self.probabilities
        ):
            start, end = section
            if start != end:
                piece = chromosome[start:end]
            else:
                piece = chromosome[start:]
            mutated_chromosome.extend(mutation.mutate(piece, probability))

        return np.array(mutated_chromosome)
