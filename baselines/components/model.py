import pytorch_lightning as L
import torch as th
from torchmetrics import MetricCollection

from modeling.losses import SparsityConstraint, ContinuityConstraint, KL_Div, JS_Div
from modeling.layers import FactorAnnealer


class HlModel(L.LightningModule):
    def __init__(
        self,
        model: th.nn.Module,
        num_classes: int,
        optimizer_class,
        val_classification_metrics: MetricCollection = None,
        val_highlight_metrics: MetricCollection = None,
        test_classification_metrics: MetricCollection = None,
        test_highlight_metrics: MetricCollection = None,
        log_metrics: bool = True,
        add_sparsity_loss: bool = False,
        sparsity_level=0.15,
        sparsity_coefficient: float = 1.00,
        mask_padding_constraints: bool = False,
        add_continuity_loss: bool = False,
        continuity_coefficient: float = 1.00,
        classification_coefficient: float = 1.00,
        skew_epochs: int = -1,
        optimizer_kwargs={},
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "model",
                "guider",
                "val_classification_metrics",
                "val_highlight_metrics",
                "test_classification_metrics",
                "test_highlight_metrics",
            ]
        )

        self.model = model
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.num_classes = num_classes
        self.log_metrics = log_metrics
        self.add_sparsity_loss = add_sparsity_loss
        self.sparsity_coefficient = sparsity_coefficient
        self.add_continuity_loss = add_continuity_loss
        self.continuity_coefficient = continuity_coefficient
        self.classification_coefficient = classification_coefficient

        self.val_classification_metrics = val_classification_metrics
        self.val_highlight_metrics = val_highlight_metrics
        self.test_classification_metrics = test_classification_metrics
        self.test_highlight_metrics = test_highlight_metrics

        self.clf_loss = th.nn.CrossEntropyLoss()
        self.sparsity_loss = SparsityConstraint(
            sparsity_level,
            mask_padding=mask_padding_constraints,
        )
        self.continuity_loss = ContinuityConstraint()

        self.store_predictions = False
        self.val_predictions = []
        self.test_predictions = []

        self.skew_pretraining = False
        self.skew_epochs = skew_epochs

    def enable_storing_predictions(self):
        self.store_predictions = True

    def disable_storing_predictions(self):
        self.store_predictions = False

    def flush_predictions(self):
        self.val_predictions.clear()
        self.test_predictions.clear()

    def forward(self, batch):
        input_ids, mask, sample_ids, y_true, highlight_true = batch
        return self.model(input_ids, mask, sample_ids)

    def compute_loss(self, y_hat, y_true, highlight_hat, attention_mask):
        total_loss = 0
        losses = {}

        clf_loss = self.clf_loss(y_hat, y_true)
        total_loss += clf_loss * self.classification_coefficient
        losses["CE"] = clf_loss

        if self.add_sparsity_loss:
            sparsity_loss = self.sparsity_loss(
                highlight_hat=highlight_hat, attention_mask=attention_mask
            )
            total_loss += sparsity_loss * self.sparsity_coefficient
            losses["SP"] = sparsity_loss

        if self.add_continuity_loss:
            continuity = self.continuity_loss(highlight_hat)
            total_loss += continuity * self.continuity_coefficient
            losses["CT"] = continuity

        return total_loss, losses

    def compute_loss_skew(self, highlight_logits, y_true):
        total_loss = 0
        losses = {}

        skew_highlight_true = th.nn.functional.one_hot(y_true, num_classes=self.num_classes)

        skew_ce = th.nn.functional.cross_entropy(
            highlight_logits[:, : self.num_classes, :].reshape(-1, 2),
            skew_highlight_true.reshape(
                -1,
            ),
        )
        total_loss += skew_ce
        losses["CE"] = skew_ce

        return total_loss, losses

    def on_train_epoch_start(self) -> None:
        if self.skew_epochs > 0:
            if self.current_epoch < self.skew_epochs:
                self.skew_pretraining = True
                if self.current_epoch == 0:
                    print("Starting skew pre-training...")
            else:
                if self.skew_pretraining:
                    print("Disabling skew pre-training. Starting normal training...")
                self.skew_pretraining = False
        else:
            self.skew_pretraining = False

    def training_step(self, batch, batch_idx):
        input_ids, mask, sample_ids, y_true, highlight_true = batch
        y_hat, highlight_hat, highlight_logits, attention_mask = self.model(
            input_ids, mask, sample_ids
        )

        if self.skew_pretraining:
            total_loss, losses = self.compute_loss_skew(
                highlight_logits=highlight_logits, y_true=y_true
            )
        else:
            total_loss, losses = self.compute_loss(
                y_hat=y_hat,
                y_true=y_true,
                highlight_hat=highlight_hat,
                attention_mask=attention_mask,
            )

        self.log(
            name="train_loss",
            value=total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        for loss_name, loss_value in losses.items():
            self.log(
                name=f"train_{loss_name}",
                value=loss_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return total_loss

    def validation_step(
        self,
        batch,
        batch_idx,
    ):
        input_ids, mask, sample_ids, y_true, highlight_true = batch
        y_hat, highlight_hat, highlight_logits, attention_mask = self.model(
            input_ids, mask, sample_ids
        )
        total_loss, losses = self.compute_loss(
            y_hat=y_hat,
            y_true=y_true,
            highlight_hat=highlight_hat,
            attention_mask=attention_mask,
        )

        self.log(
            name="val_loss",
            value=total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        for loss_name, loss_value in losses.items():
            self.log(
                name=f"val_{loss_name}",
                value=loss_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        if self.val_classification_metrics is not None:
            y_hat = th.argmax(y_hat, dim=-1)
            self.val_classification_metrics.update(y_hat, y_true)

        if self.val_highlight_metrics is not None:
            self.val_highlight_metrics.update(highlight_hat, highlight_true)

        if self.store_predictions:
            self.val_predictions.append(
                {
                    "y_true": y_true.detach().cpu(),
                    "y_hat": y_hat.detach().cpu(),
                    "hl_true": highlight_true.detach().cpu(),
                    "hl_hat": highlight_hat.detach().cpu(),
                    "input_ids": input_ids.detach().cpu(),
                    "sample_ids": sample_ids.detach().cpu(),
                }
            )

        return total_loss

    def validation_epoch_end(self, outputs) -> None:
        if self.val_classification_metrics is not None:
            metric_values = self.val_classification_metrics.compute()
            for key, value in metric_values.items():
                self.log(f"val_{key}", value, prog_bar=self.log_metrics)
            self.val_classification_metrics.reset()

        if self.val_highlight_metrics is not None:
            metric_values = self.val_highlight_metrics.compute()
            for key, value in metric_values.items():
                self.log(f"val_{key}", value, prog_bar=self.log_metrics)
            self.val_highlight_metrics.reset()

    def test_step(self, batch, batch_idx):
        input_ids, mask, sample_ids, y_true, highlight_true = batch
        y_hat, highlight_hat, highlight_logits, attention_mask = self.model(
            input_ids, mask, sample_ids
        )
        total_loss, losses = self.compute_loss(
            y_hat=y_hat,
            y_true=y_true,
            highlight_hat=highlight_hat,
            attention_mask=attention_mask,
        )

        self.log(
            name="test_loss",
            value=total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        for loss_name, loss_value in losses.items():
            self.log(
                name=f"test_{loss_name}",
                value=loss_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        if self.test_classification_metrics is not None:
            y_hat = th.argmax(y_hat, dim=-1)
            self.test_classification_metrics.update(y_hat, y_true)

        if self.test_highlight_metrics is not None:
            self.test_highlight_metrics.update(highlight_hat, highlight_true)

        if self.store_predictions:
            self.test_predictions.append(
                {
                    "y_true": y_true.detach().cpu(),
                    "y_hat": y_hat.detach().cpu(),
                    "hl_true": highlight_true.detach().cpu(),
                    "hl_hat": highlight_hat.detach().cpu(),
                    "input_ids": input_ids.detach().cpu(),
                    "sample_ids": sample_ids.detach().cpu(),
                }
            )

        return total_loss

    def test_epoch_end(self, outputs) -> None:
        if self.test_classification_metrics is not None:
            metric_values = self.test_classification_metrics.compute()
            for key, value in metric_values.items():
                self.log(f"test_{key}", value, prog_bar=self.log_metrics)
            self.test_classification_metrics.reset()

        if self.test_highlight_metrics is not None:
            metric_values = self.test_highlight_metrics.compute()
            for key, value in metric_values.items():
                self.log(f"test_{key}", value, prog_bar=self.log_metrics)
            self.test_highlight_metrics.reset()

    def configure_optimizers(self):
        return self.optimizer_class(self.model.parameters(), **self.optimizer_kwargs)


class MGRHlModel(HlModel):
    def forward(self, batch):
        input_ids, mask, sample_ids, y_true, highlight_true = batch
        return self.model.forward_one_head(input_ids, mask, sample_ids)

    def compute_loss_skew(self, highlight_logits, y_true):
        total_loss = 0
        losses = {}

        skew_highlight_true = th.nn.functional.one_hot(y_true, num_classes=self.num_classes)

        for gen_hl_logits in highlight_logits:
            skew_ce = th.nn.functional.cross_entropy(
                gen_hl_logits[:, : self.num_classes, :].reshape(-1, 2),
                skew_highlight_true.reshape(
                    -1,
                ),
            )
            total_loss += skew_ce
            losses["CE"] = losses.get("CE", 0) + skew_ce

        return total_loss, losses

    def compute_loss_all(self, y_hat, y_true, highlight_hat, attention_mask):
        total_loss = 0
        losses = {}

        for gen_y_hat, gen_hl_hat in zip(y_hat, highlight_hat):
            clf_loss = self.clf_loss(gen_y_hat, y_true)
            total_loss += clf_loss * self.classification_coefficient
            losses["CE"] = losses.get("CE", 0) + clf_loss

            if self.add_sparsity_loss:
                sparsity_loss = self.sparsity_loss(
                    highlight_hat=gen_hl_hat, attention_mask=attention_mask
                )
                total_loss += sparsity_loss * self.sparsity_coefficient
                losses["SP"] = losses.get("SP", 0) + sparsity_loss

            if self.add_continuity_loss:
                continuity_loss = self.continuity_loss(gen_hl_hat)
                total_loss += continuity_loss * self.continuity_coefficient
                losses["CT"] = losses.get("CT", 0) + continuity_loss

        return total_loss, losses

    def training_step(self, batch, batch_idx):
        input_ids, mask, sample_ids, y_true, highlight_true = batch
        y_hat, highlight_hat, highlight_logits, attention_mask = self.model(
            input_ids, mask, sample_ids
        )

        if self.skew_pretraining:
            total_loss, losses = self.compute_loss_skew(
                highlight_logits=highlight_logits, y_true=y_true
            )
        else:
            total_loss, losses = self.compute_loss_all(
                y_hat=y_hat,
                y_true=y_true,
                highlight_hat=highlight_hat,
                attention_mask=attention_mask,
            )

        self.log(
            name="train_loss",
            value=total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        for loss_name, loss_value in losses.items():
            self.log(
                name=f"train_{loss_name}",
                value=loss_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return total_loss

    def validation_step(
        self,
        batch,
        batch_idx,
    ):
        input_ids, mask, sample_ids, y_true, highlight_true = batch
        y_hat, highlight_hat, attention_mask = self.model.forward_one_head(
            input_ids, mask, sample_ids
        )
        total_loss, losses = self.compute_loss(
            y_hat=y_hat,
            y_true=y_true,
            highlight_hat=highlight_hat,
            attention_mask=attention_mask,
        )

        self.log(
            name="val_loss",
            value=total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        for loss_name, loss_value in losses.items():
            self.log(
                name=f"val_{loss_name}",
                value=loss_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        if self.val_classification_metrics is not None:
            y_hat = th.argmax(y_hat, dim=-1)
            self.val_classification_metrics.update(y_hat, y_true)

        if self.val_highlight_metrics is not None:
            self.val_highlight_metrics.update(highlight_hat, highlight_true)

        if self.store_predictions:
            self.val_predictions.append(
                {
                    "y_true": y_true.detach().cpu(),
                    "y_hat": y_hat.detach().cpu(),
                    "hl_true": highlight_true.detach().cpu(),
                    "hl_hat": highlight_hat.detach().cpu(),
                    "input_ids": input_ids.detach().cpu(),
                    "sample_ids": sample_ids.detach().cpu(),
                }
            )

        return total_loss

    def test_step(self, batch, batch_idx):
        input_ids, mask, sample_ids, y_true, highlight_true = batch
        y_hat, highlight_hat, attention_mask = self.model.forward_one_head(
            input_ids, mask, sample_ids
        )
        total_loss, losses = self.compute_loss(
            y_hat=y_hat,
            y_true=y_true,
            highlight_hat=highlight_hat,
            attention_mask=attention_mask,
        )

        self.log(
            name="test_loss",
            value=total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        for loss_name, loss_value in losses.items():
            self.log(
                name=f"test_{loss_name}",
                value=loss_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        if self.test_classification_metrics is not None:
            y_hat = th.argmax(y_hat, dim=-1)
            self.test_classification_metrics.update(y_hat, y_true)

        if self.test_highlight_metrics is not None:
            self.test_highlight_metrics.update(highlight_hat, highlight_true)

        if self.store_predictions:
            self.test_predictions.append(
                {
                    "y_true": y_true.detach().cpu(),
                    "y_hat": y_hat.detach().cpu(),
                    "hl_true": highlight_true.detach().cpu(),
                    "hl_hat": highlight_hat.detach().cpu(),
                    "input_ids": input_ids.detach().cpu(),
                    "sample_ids": sample_ids.detach().cpu(),
                }
            )

        return total_loss

    def configure_optimizers(self):
        parameters = []
        lr = self.optimizer_kwargs["lr"]
        num_generators = self.model.num_generators

        parameters.append(
            {
                "params": self.model.classification_head.parameters(),
                "lr": lr / num_generators,
            }
        )
        parameters.append(
            {
                "params": self.model.classifier_encoder.parameters(),
                "lr": lr / num_generators,
            }
        )
        for generator_idx in range(num_generators):
            parameters.append(
                {
                    "params": self.model.generators[generator_idx].parameters(),
                    "lr": lr * (generator_idx + 1),
                }
            )

        return self.optimizer_class(parameters)


class MCDHlModel(HlModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.automatic_optimization = False

        self.kl_loss = KL_Div()

    def compute_classification_loss(
        self,
        input_ids,
        attention_mask,
        sample_ids,
        y_true,
        return_predictions: bool = False,
    ):
        total_loss = 0
        losses = {}

        highlight_hat, highlight_logits = self.model.generator_forward(
            input_ids, attention_mask, sample_ids
        )

        if self.add_sparsity_loss:
            sparsity_loss = self.sparsity_loss(
                highlight_hat=highlight_hat, attention_mask=attention_mask
            )
            total_loss += sparsity_loss * self.sparsity_coefficient
            losses["SP"] = sparsity_loss

        if self.add_continuity_loss:
            continuity = self.continuity_loss(highlight_hat)
            total_loss += continuity * self.continuity_coefficient
            losses["CT"] = continuity

        forward_logits = self.model.classifier_forward(
            input_ids, attention_mask, highlight_hat.detach(), sample_ids
        )
        forward_ce = self.clf_loss(forward_logits, y_true)
        forward_loss = forward_ce * self.classification_coefficient
        total_loss += forward_loss
        losses["fwd_CE"] = forward_ce

        no_selection_logits = self.model.no_selection_forward(input_ids, attention_mask, sample_ids)
        no_selection_ce = self.clf_loss(no_selection_logits, y_true)
        total_loss += no_selection_ce * self.classification_coefficient
        losses["full_CE"] = no_selection_ce

        if return_predictions:
            return total_loss, forward_loss, losses, forward_logits, highlight_hat

        return total_loss, forward_loss, losses

    def compute_generator_loss(
        self,
        input_ids,
        attention_mask,
        sample_ids,
    ):
        total_loss = 0
        losses = {}

        highlight_hat, highlight_logits = self.model.generator_forward(
            input_ids, attention_mask, sample_ids
        )

        if self.add_sparsity_loss:
            sparsity_loss = self.sparsity_loss(
                highlight_hat=highlight_hat, attention_mask=attention_mask
            )
            total_loss += sparsity_loss * self.sparsity_coefficient
            losses["SP"] = sparsity_loss

        if self.add_continuity_loss:
            continuity = self.continuity_loss(highlight_hat)
            total_loss += continuity * self.continuity_coefficient
            losses["CT"] = continuity

        forward_logits = self.model.classifier_forward(
            input_ids, attention_mask, highlight_hat, sample_ids
        )
        no_selection_logits = self.model.no_selection_forward(input_ids, attention_mask, sample_ids)

        kl_loss = self.kl_loss(forward_logits, no_selection_logits)
        total_loss += kl_loss
        losses["KL"] = kl_loss

        return total_loss, losses

    def training_step(self, batch, batch_idx):
        input_ids, mask, sample_ids, y_true, highlight_true = batch

        optimizer_gen, optimizer_cls = self.optimizers()

        optimizer_gen.zero_grad()
        optimizer_cls.zero_grad()

        if self.skew_pretraining:
            highlight_hat, highlight_logits = self.model.generator_forward(
                input_ids, mask, sample_ids
            )
            total_loss, losses = self.compute_loss_skew(
                highlight_logits=highlight_logits, y_true=y_true
            )
            self.manual_backward(total_loss)
            optimizer_gen.step()
            optimizer_gen.zero_grad()

            self.log(
                name="train_loss",
                value=total_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            for loss_name, loss_value in losses.items():
                self.log(
                    name=f"train_{loss_name}",
                    value=loss_value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )
            return total_loss

        cls_loss, cls_fwd_ce, cls_losses = self.compute_classification_loss(
            input_ids=input_ids,
            attention_mask=mask,
            sample_ids=sample_ids,
            y_true=y_true,
        )
        self.manual_backward(cls_loss)

        optimizer_cls.step()
        optimizer_cls.zero_grad()
        optimizer_gen.step()
        optimizer_gen.zero_grad()

        cls_encoder_names = []
        for name, p in self.model.cls_encoder.named_parameters():
            if p.requires_grad:
                p.requires_grad = False
                cls_encoder_names.append(name)
        cls_head_names = []
        for name, p in self.model.classification_head.named_parameters():
            if p.requires_grad:
                p.requires_grad = False
                cls_head_names.append(name)

        gen_loss, gen_losses = self.compute_generator_loss(
            input_ids=input_ids, attention_mask=mask, sample_ids=sample_ids
        )
        self.manual_backward(gen_loss)
        optimizer_gen.step()
        optimizer_gen.zero_grad()
        optimizer_gen.zero_grad()

        for name, p in self.model.cls_encoder.named_parameters():
            if name in cls_encoder_names:
                p.requires_grad = True
        cls_encoder_names.clear()

        for name, p in self.model.classification_head.named_parameters():
            if name in cls_head_names:
                p.requires_grad = True
        cls_head_names.clear()

        return_loss = cls_fwd_ce + gen_loss
        return_losses = {"fwd_CE": cls_losses["fwd_CE"], **gen_losses}

        self.log(
            name="train_loss",
            value=return_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        for loss_name, loss_value in return_losses.items():
            self.log(
                name=f"train_{loss_name}",
                value=loss_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return return_loss

    def validation_step(
        self,
        batch,
        batch_idx,
    ):
        input_ids, mask, sample_ids, y_true, highlight_true = batch

        cls_loss, cls_fwd_ce, cls_losses, y_hat, highlight_hat = self.compute_classification_loss(
            input_ids=input_ids,
            attention_mask=mask,
            sample_ids=sample_ids,
            y_true=y_true,
            return_predictions=True,
        )
        gen_loss, gen_losses = self.compute_generator_loss(
            input_ids=input_ids, attention_mask=mask, sample_ids=sample_ids
        )

        return_loss = cls_fwd_ce + gen_loss
        return_losses = {"fwd_CE": cls_losses["fwd_CE"], **gen_losses}

        self.log(
            name="val_loss",
            value=return_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        for loss_name, loss_value in return_losses.items():
            self.log(
                name=f"val_{loss_name}",
                value=loss_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        if self.val_classification_metrics is not None:
            y_hat = th.argmax(y_hat, dim=-1)
            self.val_classification_metrics.update(y_hat, y_true)

        if self.val_highlight_metrics is not None:
            self.val_highlight_metrics.update(highlight_hat, highlight_true)

        if self.store_predictions:
            self.val_predictions.append(
                {
                    "y_true": y_true.detach().cpu(),
                    "y_hat": y_hat.detach().cpu(),
                    "hl_true": highlight_true.detach().cpu(),
                    "hl_hat": highlight_hat.detach().cpu(),
                    "input_ids": input_ids.detach().cpu(),
                    "sample_ids": sample_ids.detach().cpu(),
                }
            )

        return return_loss

    def test_step(self, batch, batch_idx):
        input_ids, mask, sample_ids, y_true, highlight_true = batch

        cls_loss, cls_fwd_ce, cls_losses, y_hat, highlight_hat = self.compute_classification_loss(
            input_ids=input_ids,
            attention_mask=mask,
            sample_ids=sample_ids,
            y_true=y_true,
            return_predictions=True,
        )
        gen_loss, gen_losses = self.compute_generator_loss(
            input_ids=input_ids, attention_mask=mask, sample_ids=sample_ids
        )

        return_loss = cls_fwd_ce + gen_loss
        return_losses = {"fwd_CE": cls_losses["fwd_CE"], **gen_losses}

        self.log(
            name="test_loss",
            value=return_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        for loss_name, loss_value in return_losses.items():
            self.log(
                name=f"test_{loss_name}",
                value=loss_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        if self.test_classification_metrics is not None:
            y_hat = th.argmax(y_hat, dim=-1)
            self.test_classification_metrics.update(y_hat, y_true)

        if self.test_highlight_metrics is not None:
            self.test_highlight_metrics.update(highlight_hat, highlight_true)

        if self.store_predictions:
            self.test_predictions.append(
                {
                    "y_true": y_true.detach().cpu(),
                    "y_hat": y_hat.detach().cpu(),
                    "hl_true": highlight_true.detach().cpu(),
                    "hl_hat": highlight_hat.detach().cpu(),
                    "input_ids": input_ids.detach().cpu(),
                    "sample_ids": sample_ids.detach().cpu(),
                }
            )

        return return_loss

    def configure_optimizers(self):
        generator_params = []
        for p in self.model.gen.parameters():
            if p.requires_grad:
                generator_params.append(p)

        optimizer_gen = self.optimizer_class(
            [{"params": generator_params, **self.optimizer_kwargs}]
        )

        classifier_params = []
        for p in self.model.cls_encoder.parameters():
            if p.requires_grad:
                classifier_params.append(p)
        for p in self.model.classification_head.parameters():
            if p.requires_grad:
                classifier_params.append(p)

        optimizer_cls = self.optimizer_class(
            [{"params": classifier_params, **self.optimizer_kwargs}]
        )
        return optimizer_gen, optimizer_cls


class GRATHlModel(HlModel):
    def __init__(
        self,
        guider: th.nn.Module,
        pretrain_epochs: int = 10,
        guide_coefficient: float = 1.00,
        jsd_coefficient: float = 1.00,
        guide_decay: float = 1e-04,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(
            ignore=[
                "model",
                "guider",
                "val_classification_metrics",
                "val_highlight_metrics",
                "test_classification_metrics",
                "test_highlight_metrics",
            ]
        )
        self.automatic_optimization = False

        self.guider = guider
        self.pretrain_epochs = pretrain_epochs
        self.guide_coefficient = guide_coefficient
        self.jsd_coefficient = jsd_coefficient
        self.guide_decay = guide_decay

        self.jsd = JS_Div()

        self.train_model = False

        def decay_func(current_step: int):
            return max(-float(current_step) * self.guide_decay + 1.0, 0.0)

        self.guide_annealer = FactorAnnealer(factor=1.0, decay_callback=decay_func)

    def forward(self, batch):
        input_ids, mask, sample_ids, y_true, highlight_true = batch
        return self.model(input_ids, mask, sample_ids)

    def compute_model_loss(
        self,
        guider_attention,
        guider_y_hat,
        y_hat,
        y_true,
        highlight_hat,
        attention_mask,
        highlight_logits,
    ):
        total_loss = 0
        losses = {}

        clf_loss = self.clf_loss(y_hat, y_true)
        total_loss += clf_loss * self.classification_coefficient
        losses["CE"] = clf_loss

        if self.add_sparsity_loss:
            sparsity_loss = self.sparsity_loss(
                highlight_hat=highlight_hat, attention_mask=attention_mask
            )
            total_loss += sparsity_loss * self.sparsity_coefficient
            losses["SP"] = sparsity_loss

        if self.add_continuity_loss:
            continuity = self.continuity_loss(highlight_hat)
            total_loss += continuity * self.continuity_coefficient
            losses["CT"] = continuity

        guider_attention = guider_attention[:, :, 0].detach()
        scaling = th.mean(guider_attention, dim=-1) + (
            1.0 / (1.0 + attention_mask.float().sum(dim=-1))
        )
        scaling_attention = guider_attention / (1e-8 + scaling.unsqueeze(dim=1))
        scaling_attention = th.clamp_max(scaling_attention, 1.0) * attention_mask.float()

        guide_bce = th.nn.functional.binary_cross_entropy_with_logits(
            highlight_logits[:, :, 1], scaling_attention
        )
        guide_bce_coefficient = self.guide_annealer.current_factor * self.guide_coefficient
        total_loss += guide_bce * guide_bce_coefficient
        losses["guide_BCE"] = guide_bce

        js_div = self.jsd(y_hat, guider_y_hat.detach())
        js_div_coefficient = self.jsd_coefficient * (1.0 - self.guide_annealer.current_factor)
        total_loss += js_div * js_div_coefficient
        losses["JS"] = js_div

        return total_loss, losses

    def compute_guider_loss(self, y_hat, y_true):
        clf_loss = self.clf_loss(y_hat, y_true)
        return clf_loss, {"guider_CE": clf_loss}

    def training_step(self, batch, batch_idx):
        guider_opt, model_opt = self.optimizers()

        input_ids, mask, sample_ids, y_true, highlight_true = batch

        if self.skew_pretraining:
            y_hat, highlight_hat, highlight_logits, attention_mask = self.model(
                input_ids, mask, sample_ids
            )
            total_loss, losses = self.compute_loss_skew(
                highlight_logits=highlight_logits, y_true=y_true
            )
            model_opt.zero_grad()
            self.manual_backward(total_loss)
            model_opt.step()
            model_opt.zero_grad()

            self.log(
                name="train_loss",
                value=total_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            for loss_name, loss_value in losses.items():
                self.log(
                    name=f"train_{loss_name}",
                    value=loss_value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )
            return total_loss

        guider_opt.zero_grad()

        guider_attention, guider_y_hat = self.guider(input_ids, mask, sample_ids)
        guider_total_loss, guider_losses = self.compute_guider_loss(
            y_hat=guider_y_hat, y_true=y_true
        )

        self.manual_backward(guider_total_loss)
        guider_opt.step()
        guider_opt.zero_grad()

        self.guider.eval()
        model_opt.zero_grad()

        y_hat, highlight_hat, highlight_logits, attention_mask = self.model(
            input_ids, mask, sample_ids
        )
        guider_attention, guider_y_hat = self.guider(input_ids, mask, sample_ids)

        total_loss, losses = self.compute_model_loss(
            y_hat=y_hat,
            y_true=y_true,
            highlight_hat=highlight_hat,
            attention_mask=attention_mask,
            guider_attention=guider_attention,
            guider_y_hat=guider_y_hat,
            highlight_logits=highlight_logits,
        )

        self.guider.train()

        if self.train_model:
            self.manual_backward(total_loss)

            model_opt.step()
            model_opt.zero_grad()

            self.guide_annealer.step()

        self.log(
            name="train_loss",
            value=total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        for loss_name, loss_value in guider_losses.items():
            self.log(
                name=f"train_{loss_name}",
                value=loss_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        for loss_name, loss_value in losses.items():
            self.log(
                name=f"train_{loss_name}",
                value=loss_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return total_loss

    def on_train_epoch_end(self) -> None:
        if self.skew_pretraining:
            return

        skew_epochs = max(self.skew_epochs, 0)
        if self.current_epoch - skew_epochs > self.pretrain_epochs and not self.train_model:
            print("Pre-training ended! Starting training model with guider...")
            self.train_model = True

    def validation_step(
        self,
        batch,
        batch_idx,
    ):
        input_ids, mask, sample_ids, y_true, highlight_true = batch

        y_hat, highlight_hat, highlight_logits, attention_mask = self.model(
            input_ids, mask, sample_ids
        )
        guider_attention, guider_y_hat = self.guider(input_ids, mask, sample_ids)

        total_loss, losses = self.compute_model_loss(
            y_hat=y_hat,
            y_true=y_true,
            highlight_hat=highlight_hat,
            attention_mask=attention_mask,
            guider_attention=guider_attention,
            guider_y_hat=guider_y_hat,
            highlight_logits=highlight_logits,
        )

        self.log(
            name="val_loss",
            value=total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        for loss_name, loss_value in losses.items():
            self.log(
                name=f"val_{loss_name}",
                value=loss_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        if self.val_classification_metrics is not None:
            y_hat = th.argmax(y_hat, dim=-1)
            self.val_classification_metrics.update(y_hat, y_true)

        if self.val_highlight_metrics is not None:
            self.val_highlight_metrics.update(highlight_hat, highlight_true)

        if self.store_predictions:
            self.val_predictions.append(
                {
                    "y_true": y_true.detach().cpu(),
                    "y_hat": y_hat.detach().cpu(),
                    "hl_true": highlight_true.detach().cpu(),
                    "hl_hat": highlight_hat.detach().cpu(),
                    "input_ids": input_ids.detach().cpu(),
                    "sample_ids": sample_ids.detach().cpu(),
                }
            )

        return total_loss

    def validation_epoch_end(self, outputs) -> None:
        if self.val_classification_metrics is not None:
            metric_values = self.val_classification_metrics.compute()
            for key, value in metric_values.items():
                self.log(f"val_{key}", value, prog_bar=self.log_metrics)
            self.val_classification_metrics.reset()

        if self.val_highlight_metrics is not None:
            metric_values = self.val_highlight_metrics.compute()
            for key, value in metric_values.items():
                self.log(f"val_{key}", value, prog_bar=self.log_metrics)
            self.val_highlight_metrics.reset()

    def test_step(self, batch, batch_idx):
        input_ids, mask, sample_ids, y_true, highlight_true = batch

        y_hat, highlight_hat, highlight_logits, attention_mask = self.model(
            input_ids, mask, sample_ids
        )
        guider_attention, guider_y_hat = self.guider(input_ids, mask, sample_ids)

        total_loss, losses = self.compute_model_loss(
            y_hat=y_hat,
            y_true=y_true,
            highlight_hat=highlight_hat,
            attention_mask=attention_mask,
            guider_attention=guider_attention,
            guider_y_hat=guider_y_hat,
            highlight_logits=highlight_logits,
        )

        self.log(
            name="test_loss",
            value=total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        for loss_name, loss_value in losses.items():
            self.log(
                name=f"test_{loss_name}",
                value=loss_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        if self.test_classification_metrics is not None:
            y_hat = th.argmax(y_hat, dim=-1)
            self.test_classification_metrics.update(y_hat, y_true)

        if self.test_highlight_metrics is not None:
            self.test_highlight_metrics.update(highlight_hat, highlight_true)

        if self.store_predictions:
            self.test_predictions.append(
                {
                    "y_true": y_true.detach().cpu(),
                    "y_hat": y_hat.detach().cpu(),
                    "hl_true": highlight_true.detach().cpu(),
                    "hl_hat": highlight_hat.detach().cpu(),
                    "input_ids": input_ids.detach().cpu(),
                    "sample_ids": sample_ids.detach().cpu(),
                }
            )

        return total_loss

    def test_epoch_end(self, outputs) -> None:
        if self.test_classification_metrics is not None:
            metric_values = self.test_classification_metrics.compute()
            for key, value in metric_values.items():
                self.log(f"test_{key}", value, prog_bar=self.log_metrics)
            self.test_classification_metrics.reset()

        if self.test_highlight_metrics is not None:
            metric_values = self.test_highlight_metrics.compute()
            for key, value in metric_values.items():
                self.log(f"test_{key}", value, prog_bar=self.log_metrics)
            self.test_highlight_metrics.reset()

    def configure_optimizers(self):
        optimizer_guider = self.optimizer_class(self.guider.parameters(), **self.optimizer_kwargs)
        optimizer_model = self.optimizer_class(self.model.parameters(), **self.optimizer_kwargs)

        return optimizer_guider, optimizer_model
