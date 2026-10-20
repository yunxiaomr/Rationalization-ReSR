import numpy as np
from math import prod
import torch


def flatten(weights: dict[str, torch.Tensor]) -> np.ndarray:

    values = list(weights.values())
    concatenation = np.concatenate(values, axis=None)

    return concatenation


def match_format(weights: np.ndarray, weights_format: dict[str, tuple]) -> dict[str, torch.Tensor]:

    result: dict[str, torch.Tensor] = dict()

    current_idx = 0
    for layer in weights_format:
        size = weights_format[layer]
        length = prod(size)
        layer_params = weights[current_idx : current_idx + length]

        shaped_params = torch.Tensor(layer_params)
        shaped_params = torch.reshape(shaped_params, size)
        result[layer] = shaped_params
        current_idx += length

    return result


def get_format(weights: dict[str, torch.Tensor]) -> dict[str, tuple]:

    shape: dict[str, tuple] = dict()

    for layer in weights:
        layer_params = weights[layer]
        size = tuple(layer_params.size())
        shape[layer] = size

    return shape


def cut_to_size(array: np.ndarray | list, size: int) -> np.ndarray:

    n_samples = len(array)
    n_samples -= n_samples % size

    return array[0:n_samples]
