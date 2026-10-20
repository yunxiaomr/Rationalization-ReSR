import numpy as np
from tqdm import tqdm


def get_texts_len(texts: np.ndarray) -> list[int]:

    lengths = []

    for text in texts:
        length = 0
        for token in text:
            if token.any():
                length += 1
            else:
                break
        lengths.append(length)

    return lengths


def intersection_over_union(y_pred: np.ndarray, y_true: np.ndarray) -> float:

    intersection = np.sum(y_pred * y_true)
    union = np.sum(y_pred + y_true) - intersection

    iou = intersection / union if union != 0.0 else np.nan

    return iou


def mirror_intersection_over_union(y_pred: np.ndarray, y_true: np.ndarray) -> float:

    mirror_y_pred = np.where(y_pred == 1.0, 0.0, 1.0)
    mirror_y_true = np.where(y_true == 1.0, 0.0, 1.0)

    return intersection_over_union(mirror_y_pred, mirror_y_true)


def binary_micro_scores(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:

    y_pred = y_pred[0 : len(y_true)]

    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    precision = tp / (tp + fp) if (tp + fp) != 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) != 0 else np.nan
    f1_score = (
        2 * recall * precision / (recall + precision) if (recall + precision) != 0 else np.nan
    )
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + fp + tn + fn) != 0 else np.nan

    scores = {
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        "accuracy": accuracy,
    }

    return scores


def binary_micro_scores_multi_group(
    y_pred: np.ndarray, y_true: list[list[np.ndarray]]
) -> dict[str, float]:

    assert len(y_pred) == len(y_true)

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for group, pred in tqdm(zip(y_true, y_pred)):
        if len(group) == 0:
            continue
        elif len(group) > 1:
            best_group = group[0]
            best_score = binary_micro_scores(pred, best_group)["f1"]
            for g in group[1:]:
                group_score = binary_micro_scores(pred, g)["f1"]
                if group_score > best_score:
                    best_score = group_score
                    best_group = g
        else:
            best_group = group[0]

        pred = pred[0 : len(best_group)]

        tp += np.sum((pred == 1) & (best_group == 1))
        tn += np.sum((pred == 0) & (best_group == 0))
        fp += np.sum((pred == 1) & (best_group == 0))
        fn += np.sum((pred == 0) & (best_group == 1))

    precision = tp / (tp + fp) if (tp + fp) != 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) != 0 else np.nan
    f1_score = (
        2 * recall * precision / (recall + precision)
        if precision != np.nan and recall != np.nan
        else np.nan
    )
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn != 0 else np.nan

    scores = {
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        "accuracy": accuracy,
    }

    return scores


def reduce_masks_by_majority(
    masks: list[list[np.ndarray]], texts_len: list[int]
) -> list[np.ndarray]:

    assert len(texts_len) == len(masks)

    new_masks = []

    for mask, text_len in zip(masks, texts_len):
        if len(mask) == 1:
            new_mask = mask[0][:text_len]

        elif len(mask) > 1:
            n_annotators = len(mask)
            min_vote = n_annotators / 2.0
            new_mask = np.array([seq[:text_len] for seq in mask])
            votes = np.sum(new_mask, axis=0)
            new_mask = np.where(votes > min_vote, 1.0, 0.0)

        else:
            new_mask = [0] * text_len

        new_masks.append(np.array(new_mask))

    return new_masks


class StatisticalMetrics:
    def __init__(self):

        self.metrics = dict()

    def add_metrics(self, metrics: dict) -> None:

        for key in metrics:
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(metrics[key])

    def compute_means(self) -> dict:

        means = dict()

        for key, values in self.metrics.items():
            means[key] = np.mean(values)

        return means

    def compute_stds(self) -> dict:

        stds = dict()

        for key, values in self.metrics.items():
            stds[key] = np.std(values)

        return stds

    def get_textual_statistics(self) -> list[str]:

        means = self.compute_means()
        stds = self.compute_stds()

        statistics: list[str] = []

        for key in means:
            mean = means[key]
            std = stds[key]
            text = str(key) + ": Mean: {}; Std: {}".format(mean, std)
            statistics.append(text)

        return statistics
