import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model.highlight_extractor import HighlightExtractor
from src.genetic.individual import Individual

from src.utils.config import (
    INDIVIDUAL_PARAMETERS,
    MASKS_SAVE_PATH,
    METRICS_SAVE_PATH,
    MODEL_PARAMETERS,
    MODEL_SAVE_PATH,
    RUN_EAGERLY,
)
from src.utils.saving import save_to_txt, save_metrics
from train import SUPPORTED_DATASETS, _build_dataset, _build_validation_split, set_random_seed

FINE_TUNING_LR = 1e-3


def parse_arguments():

    parser = argparse.ArgumentParser(description="Gen-SPP")
    parser.add_argument(
        "dataset_name",
        choices=SUPPORTED_DATASETS,
        help="Dataset used for the saved GenSPP model",
    )
    parser.add_argument("file_name", type=str, help="Name of the file to be evaluated (e.g. test_1")
    parser.add_argument("--e", type=int, help="Number of fine-tuning epochs", default=5)
    parser.add_argument(
        "--seed", type=int, default=2023, help="Seed for validation sampling and fine-tuning"
    )

    _args = parser.parse_args()

    return _args


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Model running on device:", device)

    args = parse_arguments()
    dataset_name = args.dataset_name
    file_name = args.file_name
    epochs = args.e
    set_random_seed(args.seed)
    load_path = Path(MODEL_SAVE_PATH.format(file_name) + ".pt")

    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Invalid dataset name {dataset_name!r}; choose one of {SUPPORTED_DATASETS}"
        )

    dataset = _build_dataset(dataset_name)

    train_set, test_set = dataset.get_train_and_test_sets()
    x_train, labels_train, highlight_masks_train = train_set
    x_test, labels_test, highlight_masks_test = test_set

    x_val, labels_val, _ = _build_validation_split(dataset, train_set)

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
    INDIVIDUAL_PARAMETERS["training_epochs"] = epochs
    INDIVIDUAL_PARAMETERS["learning_rate"] = FINE_TUNING_LR
    INDIVIDUAL_PARAMETERS["max_trainings"] = 1

    model = HighlightExtractor(**MODEL_PARAMETERS)

    if not RUN_EAGERLY:
        model = torch.compile(model)

    model.load_state_dict(torch.load(load_path, map_location="cpu"))

    INDIVIDUAL_PARAMETERS["model"] = model

    individual = Individual(**INDIVIDUAL_PARAMETERS)

    individual.refine()

    metrics = individual.compute_metrics(x_test, labels_test, highlight_masks_test)

    predicted_labels = individual.compute_labels(x_test)
    test_set_masks = individual.compute_masks(x_test)

    train_texts, test_texts = dataset.get_train_and_test_texts()

    save_to_txt(
        file_name,
        MASKS_SAVE_PATH,
        test_texts,
        list(test_set_masks),
        highlight_masks_test,
        predicted_classes=list(predicted_labels),
        real_classes=labels_test,
    )

    save_metrics(file_name, METRICS_SAVE_PATH, metrics)

    for key, value in metrics.items():
        print("{}: {}".format(key, value))
