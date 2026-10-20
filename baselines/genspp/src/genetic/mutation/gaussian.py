import numpy as np

from src.genetic.mutation.mutation_interface import MutationInterface


class GaussianMutation(MutationInterface):
    def __init__(self, mean: float, std: float, noise: bool):

        self.mean = mean
        self.std = std
        self.noise = noise

    def mutate(self, chromosome: np.ndarray, mutation_prob: float) -> np.ndarray:

        length = len(chromosome)

        genes_mutations_probs = np.random.uniform(low=0.0, high=1.0, size=length)
        mutations_mask = np.where(genes_mutations_probs < mutation_prob, 1.0, 0.0)

        gaussian_mutations = np.random.normal(loc=self.mean, scale=self.std, size=length)

        if self.noise:
            noise = gaussian_mutations * mutations_mask
            mutated_chromosome = chromosome + noise
        else:
            mutated_chromosome = (gaussian_mutations * mutations_mask) + (
                chromosome * (1.0 - mutations_mask)
            )

        return mutated_chromosome
