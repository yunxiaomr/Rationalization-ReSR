from pathlib import Path

from src.genetic.selection.roulette_wheel import RouletteWheelSelection
from src.genetic.crossover.one_point import OnePointCrossover
from src.genetic.mutation.gaussian import GaussianMutation
from src.genetic.mutation.split_mutator import SplitMutation
from src.genetic.survival.half_elitism import HalfElitismSurvival

from src.model.layers.encoder import EncoderType, EncoderBuilder

GENSPP_ROOT = Path(__file__).resolve().parents[2]
BASELINES_ROOT = GENSPP_ROOT.parent
BASELINES_DATA_ROOT = BASELINES_ROOT / "data"


def _path(*parts: str) -> str:
    return str(Path(*parts))


TOY_DATASET_PATHS = [_path(BASELINES_ROOT, "data", "toy", "toy_sequential_dataset_easy.pkl")]

HATE_DATASET_PATHS = [
    _path(BASELINES_ROOT, "data", "hatexplain", "train.pkl"),
    _path(BASELINES_ROOT, "data", "hatexplain", "test.pkl"),
    _path(BASELINES_ROOT, "data", "hatexplain", "val.pkl"),
]


MODEL_SAVE_PATH = _path(GENSPP_ROOT, "results", "weights", "model_{}")
GLOVE_TWITTER_PATH = _path(BASELINES_DATA_ROOT, "glove", "twitter", "glove.twitter.27B.{}d.txt")
MASKS_SAVE_PATH = _path(GENSPP_ROOT, "results", "outputs", "masks_{}.txt")
METRICS_SAVE_PATH = _path(GENSPP_ROOT, "results", "outputs", "metrics_{}.txt")


TOY_DATASET_MAX_LEN = 20
TOY_DATASET_TRAIN_PERCENTAGE = 0.8
TOY_TOKEN_EMBEDDING_DIM = 26
TOY_HIDDEN_EMBEDDING_DIM = 8
TOY_CE_EXPECTED_LOSS = 0.1


HATE_EXPLAIN_FILTER_MAX_LEN = 30
HATE_EXPLAIN_TOKEN_EMBEDDING_DIM = 25
HATE_EXPLAIN_HIDDEN_EMBEDDING_DIM = 16
HATE_EXPLAIN_CE_EXPECTED_LOSS = 0.6

N_GENERATIONS = 100
POPULATION_SIZE = 50
SELECTION_RATE = 0.5
MUTATION_PROB = 1.0
SELECTION_STRATEGY = RouletteWheelSelection
CROSSOVER_STRATEGY = OnePointCrossover
MUTATION_STRATEGY = SplitMutation(
    sections=[(0, -1), (-1, -1)],
    mutations=[
        GaussianMutation(mean=0.0, std=0.05, noise=True),
        GaussianMutation(mean=0.0, std=0.10, noise=True),
    ],
    probabilities=[MUTATION_PROB, MUTATION_PROB],
)
SURVIVAL_STRATEGY = HalfElitismSurvival
STOP_THRESHOLD = 0.01
REFINE_WITH_SGD = True


GEN_ENCODER_BUILDER = EncoderBuilder(EncoderType.RECURRENT)
CL_ENCODER_BUILDER = EncoderBuilder(EncoderType.RECURRENT)

GEN_ENCODER_PARAMS = {
    "apply_positional_encoding": False,
    "dropout": 0.0,
    "bidirectional": False,
}

CL_ENCODER_PARAMS = {
    "apply_positional_encoding": False,
    "dropout": 0.0,
    "bidirectional": False,
}

MODEL_PARAMETERS = {
    "generator_hidden_units": [],
    "classifier_hidden_units": [],
    "gen_encoder_builder": GEN_ENCODER_BUILDER,
    "cl_encoder_builder": CL_ENCODER_BUILDER,
    "gen_encoder_params": GEN_ENCODER_PARAMS,
    "cl_encoder_params": CL_ENCODER_PARAMS,
    "shared_encoder": False,
}

VAL_SET_PERCENTAGE = 0.2
RUN_EAGERLY = True


INDIVIDUAL_PARAMETERS = {
    "batch_size": 64,
    "training_epochs": 3,
    "max_trainings": 1,
    "train_generator_only": True,
    "learning_rate": 1e-2,
    "metric": "all",
    "reduce_multi_masks": True,
    "use_confidence_in_fitness": False,
}
