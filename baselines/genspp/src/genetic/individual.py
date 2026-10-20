from tqdm import tqdm
import gc
import math

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score

from src.model.highlight_extractor import HighlightExtractor
from src.model.layers.loss import (
    CategoricalCrossEntropyLoss,
    SparsityLoss,
    ContiguityLoss,
    ConfidenceLoss,
)
from src.utils.reshaping import flatten, get_format, match_format
from src.utils.metrics import (
    binary_micro_scores,
    binary_micro_scores_multi_group,
    get_texts_len,
    intersection_over_union,
    mirror_intersection_over_union,
    reduce_masks_by_majority,
)


class Individual:
    def __init__(
        self,
        model: HighlightExtractor,
        train_loader: DataLoader,
        val_loader: DataLoader,
        original_val_masks: np.ndarray,
        ce_expected_loss: float,
        learning_rate: float,
        device,
        batch_size: int = 32,
        training_epochs: int = 1,
        max_trainings: int = 2,
        train_generator_only: bool = True,
        metric: str = "all",
        reduce_multi_masks: bool = True,
        use_confidence_in_fitness: bool = True,
    ):

        assert metric in ["micro", "macro", "all"]

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.original_val_masks = original_val_masks
        self.batch_size = batch_size
        self.training_epochs = training_epochs
        self.max_trainings = max_trainings
        self.train_generator_only = train_generator_only
        self.ce_expected_loss = ce_expected_loss
        self.learning_rate = learning_rate
        self.device = device
        self.metric = metric
        self.reduce_multi_masks = reduce_multi_masks
        self.use_confidence_in_fitness = use_confidence_in_fitness

        if train_generator_only:
            weights = self.model.generator.state_dict()
        else:
            weights = self.model.state_dict()

        self.weights_format = get_format(weights)

        self.ce_loss_layer = CategoricalCrossEntropyLoss()
        self.sparsity_loss_layer = SparsityLoss()
        self.contiguity_loss_layer = ContiguityLoss()
        self.confidence_loss_layer = ConfidenceLoss()

        self.fitness_value: float | None = None
        self.ce: float | None = None
        self.sparsity: float | None = None
        self.confidence: float | None = None
        self.contiguity: float | None = None

        self.training_counter = 0

    @property
    def chromosome(self) -> np.ndarray:

        if self.train_generator_only:
            weights = self.model.generator.state_dict()
        else:
            weights = self.model.state_dict()

        return flatten(weights)

    @property
    def fitness(self) -> float:

        if self.fitness_value is None:
            self.compute_fitness()

        return self.fitness_value

    def update_chromosome(self, new_weights: np.ndarray) -> None:

        reshaped_weights = match_format(new_weights, self.weights_format)

        if self.train_generator_only:
            self.model.generator.load_state_dict(reshaped_weights)
        else:
            self.model.load_state_dict(reshaped_weights)

        self.fitness_value = None
        self.ce = None
        self.sparsity = None
        self.confidence = None
        self.contiguity = None

    def compute_fitness(self) -> None:

        self.model.to(self.device)
        self.model.eval()

        ce_loss, highlight_masks, soft_masks = self.__evaluate_model()
        sparsity_loss = self.__evaluate_sparsity(self.original_val_masks, highlight_masks)

        self.model.to("cpu")

        torch.cuda.empty_cache()
        gc.collect()

        ce_loss = float(ce_loss)
        sparsity_loss = float(sparsity_loss)
        self.ce = ce_loss
        self.sparsity = sparsity_loss

        if not math.isfinite(ce_loss) or not math.isfinite(sparsity_loss):
            self.fitness_value = 0.0
            return

        sparsity_loss = min(max(sparsity_loss, 0.0), 1.0)

        comp_sparsity = 1.0 - sparsity_loss
        comp_ce = max(1.0 - ce_loss, 0.0)
        loss = math.sqrt(comp_sparsity * comp_ce)
        loss = 1.0 - loss

        if self.use_confidence_in_fitness:
            confidence_loss = float(self.__evaluate_confidence(self.original_val_masks, soft_masks))
            self.confidence = confidence_loss
            if not math.isfinite(confidence_loss):
                self.fitness_value = 0.0
                return
            comp_confidence = 1.0 - confidence_loss
            confidence_error = abs(comp_confidence - sparsity_loss)
            loss = loss + 0.1 * confidence_error

        if not math.isfinite(loss) or loss < 0.0:
            self.fitness_value = 0.0
        elif ce_loss > self.ce_expected_loss:
            self.fitness_value = 1.0
        else:
            self.fitness_value = 1.0 / (loss + 1e-8)

    def compute_metrics(
        self,
        x: tuple[np.ndarray, np.ndarray],
        true_labels: np.ndarray,
        true_masks: list[list[np.ndarray]],
    ) -> dict[str, float]:

        print("Computing metrics...")

        texts = x[0]
        masks = x[1]

        texts_len = get_texts_len(texts)

        self.model.to(self.device)
        self.model.eval()

        classification_output, highlight_masks, soft_masks = self.predict(x)

        classification_output = np.argmax(classification_output, axis=-1)
        true_labels = np.argmax(true_labels, axis=-1)

        if self.metric == "micro":
            metrics = self.__binary_micro_metrics(
                classification_output,
                true_labels,
                highlight_masks,
                true_masks,
                texts_len,
            )
        elif self.metric == "macro":
            metrics = self.__macro_metrics(
                classification_output,
                true_labels,
                highlight_masks,
                true_masks,
                texts_len,
            )
        else:
            metrics_micro = self.__binary_micro_metrics(
                classification_output,
                true_labels,
                highlight_masks,
                true_masks,
                texts_len,
            )
            metrics_macro = self.__macro_metrics(
                classification_output,
                true_labels,
                highlight_masks,
                true_masks,
                texts_len,
            )
            metrics = metrics_micro | metrics_macro

        metrics["Sparsity"] = self.__evaluate_sparsity(masks, highlight_masks)
        metrics["Contiguity"] = self.__evaluate_contiguity(masks, highlight_masks)
        metrics["Confidence"] = self.__evaluate_confidence(masks, soft_masks)

        torch.cuda.empty_cache()
        gc.collect()

        self.model.to("cpu")

        return metrics

    def predict(self, x: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.array]:

        texts = x[0]
        masks = x[1]

        dataset = TensorDataset(torch.Tensor(texts), torch.Tensor(masks))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        classification_outputs = []
        highlight_masks = []
        soft_masks = []

        with torch.no_grad():
            for (
                text,
                mask,
            ) in loader:
                text = text.to(self.device)
                mask = mask.to(self.device)
                classification_output, highlight_mask, soft_mask = self.model(text, mask)
                classification_outputs.extend(list(classification_output.to("cpu").numpy()))
                highlight_masks.extend(list(highlight_mask.to("cpu").numpy()))
                soft_masks.extend(list(soft_mask.to("cpu").numpy()))

        return (
            np.array(classification_outputs),
            np.array(highlight_masks),
            np.array(soft_masks),
        )

    def compute_masks(self, x: tuple[np.ndarray, np.ndarray]) -> np.ndarray:

        self.model.to(self.device)
        self.model.eval()

        _, masks, _ = self.predict(x)

        torch.cuda.empty_cache()
        gc.collect()

        self.model.to("cpu")

        return masks

    def compute_labels(self, x: tuple[np.ndarray, np.ndarray]) -> np.ndarray:

        self.model.to(self.device)
        self.model.eval()

        classification_output, _, _ = self.predict(x)

        torch.cuda.empty_cache()
        gc.collect()

        self.model.to("cpu")

        return classification_output

    def refine(self) -> None:

        if self.training_counter == self.max_trainings:
            return

        self.__custom_train_loop()

        torch.cuda.empty_cache()
        gc.collect()

        self.fitness_value = None
        self.ce = None
        self.sparsity = None
        self.contiguity = None

        self.training_counter += 1

    def __custom_train_loop(self):

        loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model.to(self.device)
        self.model.train(True)

        for epoch in range(self.training_epochs):
            self.__custom_train_step(loss, optimizer)

        self.model.to("cpu")

    def __custom_train_step(self, loss_fn, optimizer):

        for texts, masks, labels in self.train_loader:
            texts = texts.to(self.device)
            masks = masks.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()

            classification_output, highlight_mask, soft_masks = self.model(texts, masks)

            loss = loss_fn(classification_output, labels)
            loss.backward()

            optimizer.step()

    def __macro_metrics(
        self,
        classification_output: np.ndarray,
        true_labels: np.ndarray,
        highlight_masks: np.ndarray,
        true_masks: list[list[np.ndarray]],
        texts_len: list[int],
    ) -> dict[str, float]:

        metrics: dict[str, float] = dict()

        classification_f1 = f1_score(
            y_pred=classification_output, y_true=true_labels, average="macro"
        )

        if self.reduce_multi_masks:
            true_masks = reduce_masks_by_majority(true_masks, texts_len)
            true_masks = [[mask] for mask in true_masks]

        f1_scores: list[float] = []
        iou_scores: list[float] = []
        mirror_iou_scores: list[float] = []

        for group_masks, predicted_mask in tqdm(zip(true_masks, highlight_masks)):
            if len(group_masks) != 0:
                best_f1 = 0.0
                best_iou = 0.0
                best_mirror_iou = 0.0

                for mask in group_masks:
                    real_cut_mask = predicted_mask[0 : len(mask)]
                    mask_f1 = f1_score(
                        y_pred=real_cut_mask,
                        y_true=mask,
                        average="macro",
                        zero_division=1.0,
                    )
                    mask_iou = intersection_over_union(y_pred=real_cut_mask, y_true=mask)
                    mask_mirror_iou = mirror_intersection_over_union(
                        y_pred=real_cut_mask, y_true=mask
                    )

                    if mask_f1 > best_f1:
                        best_f1 = mask_f1
                    if mask_iou > best_iou:
                        best_iou = mask_iou
                    if mask_mirror_iou > best_mirror_iou:
                        best_mirror_iou = mask_mirror_iou

                f1_scores.append(best_f1)
                iou_scores.append(best_iou)
                mirror_iou_scores.append(best_mirror_iou)

        highlight_f1 = np.nanmean(f1_scores)
        iou_score = np.nanmean(iou_scores)
        mirror_iou_score = np.nanmean(mirror_iou_scores)

        metrics["Macro F1 Classification"] = classification_f1
        metrics["Macro F1 Mask"] = highlight_f1
        metrics["IoU Mask"] = iou_score
        metrics["Mirror IoU Mask"] = mirror_iou_score

        return metrics

    def __binary_micro_metrics(
        self,
        classification_output: np.ndarray,
        true_labels: np.ndarray,
        highlight_masks: np.ndarray,
        true_masks: list[list[np.ndarray]],
        texts_len: list[int],
    ) -> dict[str, float]:

        metrics: dict[str, float] = dict()

        classification_metrics = binary_micro_scores(classification_output, true_labels)

        if self.reduce_multi_masks:
            true_masks = reduce_masks_by_majority(true_masks, texts_len)
            true_masks = [[mask] for mask in true_masks]

        masks_metrics = binary_micro_scores_multi_group(highlight_masks, true_masks)

        metrics["Binary Micro F1 Classification"] = classification_metrics["f1"]
        metrics["Accuracy Classification"] = classification_metrics["accuracy"]
        metrics["Binary Micro Precision Mask"] = masks_metrics["precision"]
        metrics["Binary Micro Recall Mask"] = masks_metrics["recall"]
        metrics["Binary Micro F1 Mask"] = masks_metrics["f1"]
        metrics["Accuracy Mask"] = masks_metrics["accuracy"]

        return metrics

    def __evaluate_model(self) -> tuple[float, np.ndarray, np.ndarray]:

        ce_losses = []
        highlight_masks = []
        soft_masks = []

        with torch.no_grad():
            for (
                texts,
                masks,
                labels,
            ) in self.val_loader:
                texts = texts.to(self.device)
                masks = masks.to(self.device)
                labels = labels.to(self.device)
                classification_output, highlight_mask, soft_mask = self.model(texts, masks)
                highlight_masks.extend(list(highlight_mask.to("cpu")))
                soft_masks.extend(list(soft_mask.to("cpu")))
                ce_loss = self.ce_loss_layer(labels, classification_output)
                ce_losses.append(ce_loss.to("cpu"))

        ce_loss = torch.stack([value.float() for value in ce_losses]).mean().item()
        highlight_masks = torch.stack(highlight_masks).numpy()
        soft_masks = torch.stack(soft_masks).numpy()

        return ce_loss, highlight_masks, soft_masks

    def __evaluate_sparsity(self, original_masks: np.ndarray, highlight_masks: np.ndarray) -> float:
        def to_tensor(arr):
            if isinstance(arr, np.ndarray) and arr.dtype == np.object_:
                max_len = max(x.shape[0] if hasattr(x, "shape") else len(x) for x in arr)
                float_arr = []
                for x in arr:
                    if isinstance(x, torch.Tensor):
                        x = x.cpu().numpy()

                    if len(x) < max_len:
                        x = np.pad(x, (0, max_len - len(x)), constant_values=0)
                    float_arr.append(x.astype(np.float32))
                arr = np.stack(float_arr, axis=0)
            return torch.tensor(arr, dtype=torch.float32)

        dataset = TensorDataset(to_tensor(original_masks), to_tensor(highlight_masks))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        sparsity_losses = []

        with torch.no_grad():
            for (
                original,
                pred,
            ) in loader:
                original = original.to(self.device)
                pred = pred.to(self.device)
                sparsity_loss = self.sparsity_loss_layer(original, pred)
                sparsity_losses.append(sparsity_loss.to("cpu"))

        sparsity = torch.stack([value.float() for value in sparsity_losses]).mean().item()

        return sparsity

    def __evaluate_contiguity(
        self, original_masks: np.ndarray, highlight_masks: np.ndarray
    ) -> float:

        dataset = TensorDataset(torch.Tensor(original_masks), torch.Tensor(highlight_masks))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        contiguity_losses = []

        with torch.no_grad():
            for (
                original,
                pred,
            ) in loader:
                original = original.to(self.device)
                pred = pred.to(self.device)
                contiguity_loss = self.contiguity_loss_layer(original, pred)
                contiguity_losses.append(contiguity_loss.to("cpu"))

        contiguity = torch.stack([value.float() for value in contiguity_losses]).mean().item()

        return contiguity

    def __evaluate_confidence(self, original_masks: np.ndarray, soft_masks: np.ndarray) -> float:

        dataset = TensorDataset(torch.Tensor(original_masks), torch.Tensor(soft_masks))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        confidence_losses = []

        with torch.no_grad():
            for (
                original,
                pred,
            ) in loader:
                original = original.to(self.device)
                pred = pred.to(self.device)
                confidence_loss = self.confidence_loss_layer(original, pred)
                confidence_losses.append(confidence_loss.to("cpu"))

        confidence = torch.stack([value.float() for value in confidence_losses]).mean().item()

        return confidence
