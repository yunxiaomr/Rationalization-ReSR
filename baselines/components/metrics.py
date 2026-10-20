import numpy as np
import torch as th
from torchmetrics.metric import Metric


class BinaryHighlightF1Score(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False
    plot_lower_bound = 0.0
    plot_upper_bound = 1.0

    def __init__(self, pos_label=1, **kwargs):
        super().__init__(**kwargs)

        self.pos_label = pos_label
        self.add_state(name="tp", default=th.tensor(0, dtype=th.float), dist_reduce_fx="sum")
        self.add_state(name="fp", default=th.tensor(0, dtype=th.float), dist_reduce_fx="sum")
        self.add_state(name="fn", default=th.tensor(0, dtype=th.float), dist_reduce_fx="sum")

    def compute_target_score(
        self,
        highlight_hat,
        highlight_true,
    ):
        valid_indexes = np.where(highlight_true != -1)[0]
        highlight_hat = highlight_hat[valid_indexes]
        highlight_true = highlight_true[valid_indexes]

        tp = ((highlight_hat == self.pos_label) & (highlight_true == self.pos_label)).sum()
        fp = ((highlight_hat == self.pos_label) & (highlight_true == (1 - self.pos_label))).sum()
        fn = ((highlight_hat == (1 - self.pos_label)) & (highlight_true == self.pos_label)).sum()

        return {"tp": tp, "fp": fp, "fn": fn}

    def update(self, preds: th.Tensor, target: th.Tensor) -> None:
        preds = preds.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        for sample_idx in range(preds.shape[0]):
            sample_pred = preds[sample_idx]
            sample_target = target[sample_idx]

            sample_metrics = self.compute_target_score(
                highlight_hat=sample_pred, highlight_true=sample_target
            )

            self.tp += sample_metrics["tp"]
            self.fp += sample_metrics["fp"]
            self.fn += sample_metrics["fn"]

    def compute(self):
        f1 = (2 * self.tp) / (2 * self.tp + self.fp + self.fn)
        return f1


class SelectionRate(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state(name="rate", default=th.tensor(0, dtype=th.float), dist_reduce_fx="sum")
        self.add_state(name="samples", default=th.tensor(0, dtype=th.float), dist_reduce_fx="sum")

    def update(self, preds: th.Tensor, target: th.Tensor) -> None:
        for sample_pred, sample_target in zip(preds, target):
            valid_indexes = th.where(sample_target != -1)[0]
            valid_pred = sample_pred[valid_indexes]

            if len(valid_pred):
                sample_rate = valid_pred.mean().detach().cpu()
                self.rate += sample_rate
                self.samples += 1

    def compute(self):
        return self.rate / self.samples if self.samples > 0 else 0.0


class SelectionSize(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state(name="size", default=th.tensor(0, dtype=th.float), dist_reduce_fx="sum")
        self.add_state(name="samples", default=th.tensor(0, dtype=th.float), dist_reduce_fx="sum")

    def update(self, preds: th.Tensor, target: th.Tensor) -> None:
        for sample_pred, sample_target in zip(preds, target):
            valid_indexes = th.where(sample_target != -1)[0]
            valid_pred = sample_pred[valid_indexes]

            if len(valid_pred):
                self.size += valid_pred.sum().detach().cpu()
                self.samples += 1

    def compute(self):
        return self.size / self.samples if self.samples > 0 else 0.0
