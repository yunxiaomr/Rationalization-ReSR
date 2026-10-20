import numpy as np
import random
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.trainer import GeneticTrainer
from src.preprocessing.dataset_reader.dataset_reader_interface import (
    DatasetReaderInterface,
)
from src.preprocessing.dataset_reader.toy_dataset_reader import ToyDatasetReader
from src.preprocessing.dataset_reader.hatexplain_dataset_reader import (
    HatexplainDatasetReader,
)

from src.utils.saving import save_to_txt, save_metrics
from src.utils.config import (
    CL_ENCODER_PARAMS,
    CROSSOVER_STRATEGY,
    GEN_ENCODER_PARAMS,
    GLOVE_TWITTER_PATH,
    HATE_DATASET_PATHS,
    HATE_EXPLAIN_CE_EXPECTED_LOSS,
    HATE_EXPLAIN_FILTER_MAX_LEN,
    HATE_EXPLAIN_HIDDEN_EMBEDDING_DIM,
    HATE_EXPLAIN_TOKEN_EMBEDDING_DIM,
    INDIVIDUAL_PARAMETERS,
    MASKS_SAVE_PATH,
    METRICS_SAVE_PATH,
    MODEL_PARAMETERS,
    MODEL_SAVE_PATH,
    MUTATION_PROB,
    MUTATION_STRATEGY,
    N_GENERATIONS,
    POPULATION_SIZE,
    REFINE_WITH_SGD,
    RUN_EAGERLY,
    SELECTION_RATE,
    SELECTION_STRATEGY,
    STOP_THRESHOLD,
    SURVIVAL_STRATEGY,
    TOY_CE_EXPECTED_LOSS,
    TOY_DATASET_MAX_LEN,
    TOY_DATASET_PATHS,
    TOY_DATASET_TRAIN_PERCENTAGE,
    TOY_HIDDEN_EMBEDDING_DIM,
    TOY_TOKEN_EMBEDDING_DIM,
    VAL_SET_PERCENTAGE,
)
from src.utils.metrics import StatisticalMetrics


def set_random_seed(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_arguments():

    warning = "WARNING: w > 1 can only affect training on CPU"

    parser = argparse.ArgumentParser(description="Gen-SPP")
    parser.add_argument("dataset_name", choices=("toy", "hatexplain"), help="Dataset name")
    parser.add_argument(
        "test_name",
        type=str,
        help="Name associated to the current test",
        default="test",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[2023, 1337, 12345, 31415, 42],
        help="Random seeds; defaults to the five main-experiment seeds.",
    )
    parser.add_argument(
        "--w",
        type=int,
        help="Number of workers to train in parallel." + warning,
        default=1,
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Whether to run on CPU or GPU (if available)"
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=N_GENERATIONS,
        help="Genetic generations; keep the default for formal experiments.",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=POPULATION_SIZE,
        help="Population size; keep the default for formal experiments.",
    )

    _args = parser.parse_args()

    return _args


SUPPORTED_DATASETS = ("toy", "hatexplain")


def _build_dataset(dataset_name: str):
    if dataset_name == "toy":
        dataset_reader: DatasetReaderInterface = ToyDatasetReader(
            TOY_DATASET_MAX_LEN, TOY_DATASET_TRAIN_PERCENTAGE
        )
        dataset = dataset_reader.read(TOY_DATASET_PATHS)
        embedding_dim = TOY_TOKEN_EMBEDDING_DIM
        hidden_dim = TOY_HIDDEN_EMBEDDING_DIM
        ce_expected_loss = TOY_CE_EXPECTED_LOSS
        batch_size = 64
    elif dataset_name == "hatexplain":
        dataset_reader = HatexplainDatasetReader(
            GLOVE_TWITTER_PATH,
            HATE_EXPLAIN_TOKEN_EMBEDDING_DIM,
            HATE_EXPLAIN_FILTER_MAX_LEN,
        )
        dataset = dataset_reader.read(HATE_DATASET_PATHS)
        embedding_dim = HATE_EXPLAIN_TOKEN_EMBEDDING_DIM
        hidden_dim = HATE_EXPLAIN_HIDDEN_EMBEDDING_DIM
        ce_expected_loss = HATE_EXPLAIN_CE_EXPECTED_LOSS
        batch_size = 64
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    GEN_ENCODER_PARAMS["initial_embedding_size"] = embedding_dim
    GEN_ENCODER_PARAMS["embedding_size"] = hidden_dim
    CL_ENCODER_PARAMS["initial_embedding_size"] = embedding_dim
    CL_ENCODER_PARAMS["embedding_size"] = hidden_dim
    INDIVIDUAL_PARAMETERS["ce_expected_loss"] = ce_expected_loss
    INDIVIDUAL_PARAMETERS["batch_size"] = batch_size
    return dataset


def _build_validation_split(dataset, train_set):
    _, val_set, _ = dataset.get_train_val_test_sets()
    if val_set is not None:
        return val_set

    x_train, labels_train, _ = train_set
    val_samples = int(len(labels_train) * VAL_SET_PERCENTAGE)
    permutation = np.random.permutation(len(labels_train))
    x_val_text = np.take(x_train[0], permutation, axis=0)[0:val_samples]
    x_val_mask = np.take(x_train[1], permutation, axis=0)[0:val_samples]
    labels_val = np.take(labels_train, permutation, axis=0)[0:val_samples]
    return ([x_val_text, x_val_mask], labels_val, [])


def run_training(
    dataset_name: str,
    test_name: str,
    workers: int,
    run_id: str,
    device,
    plot_train_loss=True,
    generations: int = N_GENERATIONS,
    population_size: int = POPULATION_SIZE,
) -> dict:

    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Invalid dataset name {dataset_name!r}; choose one of {SUPPORTED_DATASETS}"
        )

    run_name = f"{test_name}_{run_id}"
    dataset = _build_dataset(dataset_name)

    train_set, test_set = dataset.get_train_and_test_sets()

    x_train, labels_train, highlight_masks_train = train_set
    x_test, labels_test, highlight_masks_test = test_set

    val_set = _build_validation_split(dataset, train_set)
    x_val, labels_val, _ = val_set

    dataset_train = TensorDataset(
        torch.Tensor(x_train[0]), torch.Tensor(x_train[1]), torch.Tensor(labels_train)
    )
    train_loader = DataLoader(
        dataset_train, batch_size=INDIVIDUAL_PARAMETERS["batch_size"], shuffle=True
    )

    dataset_val = TensorDataset(
        torch.Tensor(x_val[0]), torch.Tensor(x_val[1]), torch.Tensor(labels_val)
    )
    val_loader = DataLoader(
        dataset_val, batch_size=INDIVIDUAL_PARAMETERS["batch_size"], shuffle=False
    )

    MODEL_PARAMETERS["n_classes"] = dataset.get_classes_count()
    MODEL_PARAMETERS["gen_encoder_params"]["max_len"] = dataset.max_len
    MODEL_PARAMETERS["cl_encoder_params"]["max_len"] = dataset.max_len
    INDIVIDUAL_PARAMETERS["train_loader"] = train_loader
    INDIVIDUAL_PARAMETERS["val_loader"] = val_loader
    INDIVIDUAL_PARAMETERS["original_val_masks"] = x_val[1]
    INDIVIDUAL_PARAMETERS["device"] = device

    trainer = GeneticTrainer(
        generations,
        population_size,
        SELECTION_RATE,
        MUTATION_PROB,
        SELECTION_STRATEGY,
        CROSSOVER_STRATEGY,
        MUTATION_STRATEGY,
        SURVIVAL_STRATEGY,
        MODEL_PARAMETERS,
        INDIVIDUAL_PARAMETERS,
        dataset.embedding_dim,
        dataset.max_len,
        RUN_EAGERLY,
        INDIVIDUAL_PARAMETERS["train_generator_only"],
        STOP_THRESHOLD,
        REFINE_WITH_SGD,
        workers,
    )

    trainer.train(plot_results=plot_train_loss)

    trainer.save_best_individual(run_name, MODEL_SAVE_PATH)

    train_set_metrics = trainer.get_metrics(x_train, labels_train, highlight_masks_train)
    test_set_metrics = trainer.get_metrics(x_test, labels_test, highlight_masks_test)

    test_set_masks = trainer.get_masks(x_test)
    valid_mask = np.asarray(x_test[1], dtype=np.float32)
    selected_mask = np.asarray(test_set_masks, dtype=np.float32) * valid_mask
    valid_lengths = valid_mask.sum(axis=-1)
    selected_sizes = selected_mask.sum(axis=-1)
    test_set_metrics["Selection Rate"] = np.mean(selected_sizes / np.maximum(valid_lengths, 1.0))
    test_set_metrics["Selection Size"] = np.mean(selected_sizes)
    train_texts, test_texts = dataset.get_train_and_test_texts()

    print()
    print("Metrics - Training set:")
    print(train_set_metrics)
    print()

    print("Metrics - Test set:")
    print(test_set_metrics)
    print()

    predicted_labels = trainer.get_labels(x_test)

    save_to_txt(
        run_name,
        MASKS_SAVE_PATH,
        test_texts,
        list(test_set_masks),
        highlight_masks_test,
        predicted_classes=list(predicted_labels),
        real_classes=labels_test,
    )

    save_metrics(run_name, METRICS_SAVE_PATH, test_set_metrics)

    return test_set_metrics


def _json_float(value):
    if isinstance(value, np.generic):
        return value.item()
    return float(value)


def save_seed_summary(
    test_name: str,
    dataset_name: str,
    records: list[dict],
    generations: int,
    population_size: int,
) -> None:
    output_path = Path(METRICS_SAVE_PATH.format(f"summary_{test_name}")).with_suffix(".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset_name,
        "test_name": test_name,
        "seeds": [record["seed"] for record in records],
        "generations": int(generations),
        "population_size": int(population_size),
        "records": records,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    args = parse_arguments()
    _dataset_name = args.dataset_name
    _test_name = args.test_name
    _workers = args.w
    use_cpu = args.cpu
    generations = int(args.generations)
    population_size = int(args.population_size)
    budget_tags = []
    if generations != N_GENERATIONS:
        budget_tags.append(f"generations{generations}")
    if population_size != POPULATION_SIZE:
        budget_tags.append(f"population{population_size}")
    if budget_tags:
        _test_name += "__" + "_".join(budget_tags)

    if use_cpu:
        dev = torch.device("cpu")
        if _workers > 1:
            RUN_EAGERLY = True
    else:
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            _workers = 1
        else:
            dev = torch.device("cpu")
            if _workers > 1:
                RUN_EAGERLY = True

    print("Model running on device:", dev)

    statistical_metrics = StatisticalMetrics()

    assert _workers >= 1
    assert generations >= 1
    assert population_size >= 2

    selected_seeds = list(args.seeds)
    if len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError(f"Duplicate seeds are not allowed: {selected_seeds}")

    seed_records = []
    for run_index, seed in enumerate(selected_seeds, start=1):
        print("RUNNING TRAINING {}/{} (seed={})".format(run_index, len(selected_seeds), seed))
        set_random_seed(seed)
        run_id = f"seed={seed}"
        metrics = run_training(
            _dataset_name,
            _test_name,
            _workers,
            run_id,
            dev,
            plot_train_loss=False,
            generations=generations,
            population_size=population_size,
        )
        statistical_metrics.add_metrics(metrics)
        seed_records.append(
            {
                "seed": int(seed),
                "metrics": {key: _json_float(value) for key, value in metrics.items()},
            }
        )
        save_seed_summary(
            _test_name,
            _dataset_name,
            seed_records,
            generations=generations,
            population_size=population_size,
        )

    statistics = statistical_metrics.get_textual_statistics()

    for line in statistics:
        print(line)
