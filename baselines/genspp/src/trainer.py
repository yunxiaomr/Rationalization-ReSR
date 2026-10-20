import gc
import time
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

from typing import Type
import numpy as np
import matplotlib.pyplot as plt

import torch

from src.genetic.selection.selection_interface import SelectionInterface
from src.genetic.crossover.crossover_interface import CrossoverInterface
from src.genetic.mutation.mutation_interface import MutationInterface

from src.genetic.survival.survival_interface import SurvivalInterface

from src.genetic.individual import Individual
from src.genetic.fitness_utils import safe_fitness
from src.model.highlight_extractor import HighlightExtractor


class GeneticTrainer:
    def __init__(
        self,
        n_generations: int,
        population_size: int,
        selection_rate: float,
        mutation_prob: float,
        selection_strategy: Type[SelectionInterface],
        crossover_strategy: Type[CrossoverInterface],
        mutation_strategy: MutationInterface,
        survival_strategy: Type[SurvivalInterface],
        model_params: dict,
        individual_params: dict,
        token_embedding_dim: int,
        max_len: int,
        run_eagerly: bool,
        train_generator_only: bool,
        stop_threshold: float = 0.05,
        refine: bool = True,
        workers: int = 1,
    ):

        assert 0.0 < selection_rate <= 1.0
        assert 0.0 < mutation_prob <= 1.0

        self.n_generations = n_generations
        self.population_size = population_size
        self.selection_rate = selection_rate
        self.mutation_prob = mutation_prob
        self.stop_threshold = stop_threshold
        self.refine = refine
        self.train_generator_only = train_generator_only
        self.workers = workers

        self.selection_strategy = selection_strategy
        self.crossover_strategy = crossover_strategy
        self.mutation_strategy = mutation_strategy
        self.survival_strategy = survival_strategy

        self.model_params = model_params
        self.individual_params = individual_params
        self.run_eagerly = run_eagerly
        self.max_len = max_len
        self.token_embedding_dim = token_embedding_dim

        self.population: list[Individual] = []

        self.current_generation = 0
        self.training_progress = []

        self.max_population_size = population_size + 2 * int(selection_rate * population_size)
        self.models_pool, self.initial_weights = self.__build_models_pool()

    def initialize(self) -> None:

        self.population.clear()

        for model in self.models_pool:
            self.models_pool[model] = None

        for _ in range(self.population_size):
            individual = self.__create_individual()
            self.population.append(individual)

        self.current_generation = 0
        self.training_progress.clear()

    def train(self, plot_results: bool = True) -> None:

        self.initialize()

        n_params = sum(p.numel() for p in self.population[0].model.parameters() if p.requires_grad)
        print("Trainable variables: {}".format(n_params))

        if self.refine:
            self.__apply_sgd_refinement()

        print("Training started for {} generations".format(self.n_generations))

        for _ in range(self.n_generations):
            try:
                start_time = time.time()
                self.__run_generation()
                self.current_generation += 1
                best_individual = self.get_best_individual()
                best_fitness = safe_fitness(best_individual.fitness)
                best_loss = float("inf") if best_fitness <= 0.0 else 1.0 / best_fitness

                end_time = time.time()
                print(
                    "Generation {}/{} completed in {}s -> ".format(
                        self.current_generation,
                        self.n_generations,
                        int(end_time - start_time),
                    )
                )
                print("Lowest loss: {}".format(best_loss))
                print(
                    "Cross-Entropy: {} - Sparsity: {} - Confidence: {} - Contiguity: {}".format(
                        best_individual.ce,
                        best_individual.sparsity,
                        best_individual.confidence,
                        best_individual.contiguity,
                    )
                )

                self.training_progress.append(best_loss)
                if best_loss <= self.stop_threshold:
                    break
            except KeyboardInterrupt:
                break

        if plot_results:
            self.plot_training_progress()

    def plot_training_progress(self) -> None:

        n_samples = len(self.training_progress)

        if n_samples == 0:
            raise Exception("train must be called before plotting")

        x = np.arange(1, n_samples + 1)
        y = np.array(self.training_progress)

        plt.xlabel("Generations")
        plt.ylabel("Score")
        plt.ylim(0.0, 1.0)

        plt.plot(x, y)
        plt.show()

    def get_metrics(
        self,
        x: tuple[np.ndarray, np.ndarray],
        true_labels: np.ndarray,
        true_masks: list[list[np.ndarray]],
    ) -> dict[str, float]:

        best_individual = self.get_best_individual()
        return best_individual.compute_metrics(x, true_labels, true_masks)

    def get_masks(self, x: tuple[np.ndarray, np.ndarray]) -> np.ndarray:

        best_individual = self.get_best_individual()
        return best_individual.compute_masks(x)

    def get_labels(self, x: tuple[np.ndarray, np.ndarray]) -> np.ndarray:

        best_individual = self.get_best_individual()
        return best_individual.compute_labels(x)

    def get_best_individual(self) -> Individual:

        if not self.population:
            raise RuntimeError("Cannot select a best individual from an empty population.")
        return max(self.population, key=lambda individual: safe_fitness(individual.fitness))

    def save_best_individual(self, name: str, path: str) -> None:

        best_individual = self.get_best_individual()
        best_model = best_individual.model

        save_path = path.format(name) + ".pt"
        import os

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(best_model.state_dict(), save_path)

    def __build_models_pool(
        self,
    ) -> tuple[dict[HighlightExtractor, Individual | None], dict[HighlightExtractor, dict]]:

        pool: dict[HighlightExtractor, Individual | None] = dict()
        initial_weights: dict[HighlightExtractor, dict] = dict()

        print("Building models...")

        for _ in tqdm(range(self.max_population_size)):
            model = HighlightExtractor(**self.model_params)

            if not self.run_eagerly:
                model = torch.compile(model)

            gc.collect()

            pool[model] = None
            if self.train_generator_only:
                initial_weights[model] = model.classifier.state_dict()
            else:
                initial_weights[model] = model.state_dict()

        return pool, initial_weights

    def __allocate_new_model(self) -> HighlightExtractor:

        print(
            "WARNING: Allocating new model: repeating this operation many times could cause a memory leak"
        )

        model = HighlightExtractor(**self.model_params)

        if not self.run_eagerly:
            model = torch.compile(model)

        gc.collect()

        self.models_pool[model] = None
        if self.train_generator_only:
            self.initial_weights[model] = model.classifier.state_dict()
        else:
            self.initial_weights[model] = model.state_dict()

        return model

    def __create_individual(self, chromosome: np.ndarray | None = None) -> Individual:

        model = self.__get_free_model()

        if model is None:
            model = self.__allocate_new_model()
        else:
            weights = self.initial_weights[model]
            if self.train_generator_only:
                model.classifier.load_state_dict(weights)
            else:
                model.load_state_dict(weights)

        self.individual_params["model"] = model
        individual = Individual(**self.individual_params)
        self.models_pool[model] = individual

        if chromosome is not None:
            individual.update_chromosome(chromosome)

        return individual

    def __get_free_model(self) -> HighlightExtractor | None:

        for model in self.models_pool:
            if self.models_pool[model] is None:
                return model

        return None

    def __get_model_from_individual(self, individual: Individual) -> HighlightExtractor | None:

        for model in self.models_pool:
            if self.models_pool[model] == individual:
                return model

        return None

    def __apply_sgd_refinement(self) -> None:

        print("Refining models...")

        if self.workers == 1:
            for individual in tqdm(self.population):
                individual.refine()
                gc.collect()

        else:
            with ThreadPool(processes=self.workers) as pool:
                pool.map(lambda x: x.refine(), self.population)
            gc.collect()

    def __remove_extra_individuals(self) -> None:

        n_extra = self.max_population_size - self.population_size

        for _ in range(n_extra):
            individual = self.population[-1]
            model = self.__get_model_from_individual(individual)
            if model is None:
                raise Exception("No model for the individual")
            self.models_pool[model] = None
            del self.population[-1]

        assert len(self.population) == self.population_size

        gc.collect()

    def __run_generation(self) -> None:

        parents = self.selection_strategy.select(
            self.population, self.selection_rate, workers=self.workers
        )

        for couple in tqdm(parents):
            parent_1 = couple[0]
            parent_2 = couple[1]
            child_chromosome_1, child_chromosome_2 = self.crossover_strategy.apply_crossover(
                parent_1, parent_2
            )

            child_chromosome_1 = self.mutation_strategy.mutate(
                child_chromosome_1, self.mutation_prob
            )
            child_chromosome_2 = self.mutation_strategy.mutate(
                child_chromosome_2, self.mutation_prob
            )

            child_1 = self.__create_individual(chromosome=child_chromosome_1)
            child_2 = self.__create_individual(chromosome=child_chromosome_2)

            self.population.append(child_1)
            self.population.append(child_2)

        if self.refine:
            self.__apply_sgd_refinement()

        self.population = self.survival_strategy.survival_select(
            self.population, self.population_size, workers=self.workers
        )
        self.__remove_extra_individuals()
