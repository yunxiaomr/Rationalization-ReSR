from __future__ import annotations

import copy
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINES_ROOT = PROJECT_ROOT.joinpath("baselines")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINES_ROOT))

import pytorch_lightning as L
import torch as th
import torch.nn.functional as F
from torchmetrics import MetricCollection

from modeling.losses import SparsityConstraint, ContinuityConstraint
from resr.modeling.vrsr import VRSRHyperparams, VRSRScheduler
from resr.modeling.stability import (
    collapse_rate_recovery_loss,
    deterministic_budget_mask,
    mask_rates,
    structural_checkpoint_loss,
)


def _reset_module_parameters(module: th.nn.Module):

    for m in module.modules():
        reset_fn = getattr(m, "reset_parameters", None)
        if callable(reset_fn):
            reset_fn()


def _mean_true_label_margin(
    logits: th.Tensor,
    y_true: th.Tensor,
) -> th.Tensor:

    true_logits = logits.gather(dim=1, index=y_true.view(-1, 1)).squeeze(1)
    competitor_logits = logits.masked_fill(
        F.one_hot(y_true, num_classes=logits.shape[-1]).to(dtype=th.bool),
        float("-inf"),
    )
    max_competitor = competitor_logits.max(dim=1).values
    return (true_logits - max_competitor).mean()


def _temperature_kl_from_logits(
    student_logits: th.Tensor,
    teacher_logits: th.Tensor,
    temperature: float = 1.0,
) -> th.Tensor:

    temperature = max(float(temperature), 1e-6)
    teacher_probs = F.softmax(teacher_logits.detach() / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature**2)


class ProbePredictor(th.nn.Module):
    def __init__(
        self,
        embedding: th.nn.Module,
        cls_encoder: th.nn.Module,
        classification_head: th.nn.Module,
        dropout_p: float,
    ):

        super().__init__()
        self.embedding = embedding
        self.cls_encoder = cls_encoder
        self.classification_head = classification_head
        self.dropout = th.nn.Dropout(p=dropout_p)

    def forward(
        self,
        input_ids,
        attention_mask,
        highlight_mask,
    ):

        attention_mask = attention_mask.to(dtype=th.float32)
        highlight_mask = highlight_mask.to(dtype=th.float32)

        hl_tokens_emb = self.embedding(input_ids) * attention_mask[:, :, None]
        hl_tokens_emb *= highlight_mask[:, :, None]

        hl_tokens_emb, _ = self.cls_encoder(hl_tokens_emb)
        hl_tokens_emb = hl_tokens_emb * attention_mask[:, :, None] + (
            1.0 - attention_mask[:, :, None]
        ) * (-1e6)
        hl_tokens_emb = th.transpose(hl_tokens_emb, 1, 2)

        hl_emb, _ = th.max(hl_tokens_emb, dim=2)
        hl_emb = self.dropout(hl_emb)

        return self.classification_head(hl_emb)

    def teacher_guidance_forward(
        self,
        input_ids,
        attention_mask,
        target_labels=None,
        method: str = "grad_input",
    ):

        if method != "grad_input":
            raise ValueError(f"Reference guidance supports only `grad_input`; received {method!r}.")

        attention_mask = attention_mask.to(dtype=th.float32)
        token_embeddings = self.embedding(input_ids) * attention_mask[:, :, None]
        teacher_embeddings = token_embeddings.detach().clone().requires_grad_(True)
        hidden_states, _ = self.cls_encoder(teacher_embeddings)
        masked_hidden_states = hidden_states * attention_mask[:, :, None] + (
            1.0 - attention_mask[:, :, None]
        ) * (-1e6)
        pooled_hidden_states, _ = th.max(masked_hidden_states, dim=1)
        logits = self.classification_head(pooled_hidden_states)

        if target_labels is None:
            target_labels = th.argmax(logits, dim=-1)
        target_labels = target_labels.to(dtype=th.long)
        target_logit = logits.gather(dim=-1, index=target_labels[:, None]).sum()
        token_gradients = th.autograd.grad(
            outputs=target_logit,
            inputs=teacher_embeddings,
            retain_graph=False,
            create_graph=False,
        )[0]
        token_scores = (token_gradients * teacher_embeddings).sum(dim=-1).abs()
        valid_count = attention_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        score_mean = (token_scores * attention_mask).sum(dim=-1, keepdim=True) / valid_count
        centered_scores = (token_scores - score_mean) * attention_mask
        score_variance = centered_scores.square().sum(dim=-1, keepdim=True) / valid_count
        token_scores = centered_scores / score_variance.sqrt().clamp_min(1e-6)
        token_scores = token_scores * attention_mask
        return logits, token_scores


class ReSRHlModel(L.LightningModule):
    def __init__(
        self,
        model: th.nn.Module,
        num_classes: int,
        optimizer_class,
        optimizer_kwargs: dict,
        vrsr_hparams: VRSRHyperparams,
        train_dataloader=None,
        val_dataloader=None,
        use_probe_diagnostics: bool = True,
        use_reference_diagnostics: bool = True,
        probe_train_batches: int = 10,
        probe_train_epochs: int = 1,
        probe_lr: float = 1e-3,
        probe_init_from_online: bool = True,
        probe_calibration_coefficient: float = 0.0,
        probe_calibration_temperature: float = 1.0,
        probe_calibration_warmup_epochs: int = 0,
        probe_calibration_ramp_epochs: int = 1,
        probe_calibration_min_gain: float = 0.0,
        ref_train_batches: int = 100,
        ref_train_epochs: int = 3,
        ref_lr: float = 1e-3,
        add_sparsity_loss: bool = True,
        sparsity_level: float = 0.15,
        sparsity_coefficient: float = 1.0,
        add_continuity_loss: bool = True,
        continuity_coefficient: float = 1.0,
        drift_coefficient: float = 0.0,
        training_semantics: str = "stabilized",
        manual_gradient_clip_val: float = 1.0,
        deterministic_budget_constraints: bool = True,
        per_example_budget: bool = True,
        warmup_full_input: bool = True,
        mask_padding_constraints: bool = True,
        isolate_diagnostic_rng: bool = True,
        collapse_recovery_budget: bool = True,
        isolate_inactive_dropout: bool = True,
        diagnostic_seed: int = 15000,
        classification_coefficient: float = 1.0,
        model_selection_rate_weight: float = 2.0,
        model_selection_frag_weight: float = 0.5,
        model_selection_min_f1: float = 0.0,
        model_selection_min_rate: float = 0.0,
        model_selection_clf_penalty: float = 0.0,
        teacher_alignment_coefficient: float = 0.0,
        teacher_alignment_source: str = "online",
        teacher_alignment_mode: str = "margin",
        teacher_alignment_schedule: str = "constant",
        teacher_alignment_warmup_epochs: int = 0,
        teacher_alignment_confidence_threshold: float = 0.0,
        teacher_alignment_require_correct: bool = False,
        val_classification_metrics: MetricCollection = None,
        val_highlight_metrics: MetricCollection = None,
        test_classification_metrics: MetricCollection = None,
        test_highlight_metrics: MetricCollection = None,
        log_metrics: bool = True,
    ):

        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "model",
                "train_dataloader",
                "val_dataloader",
                "val_classification_metrics",
                "val_highlight_metrics",
                "test_classification_metrics",
                "test_highlight_metrics",
            ]
        )

        self.model = model
        self.num_classes = num_classes
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs or {}

        self.val_classification_metrics = val_classification_metrics
        self.val_highlight_metrics = val_highlight_metrics
        self.test_classification_metrics = test_classification_metrics
        self.test_highlight_metrics = test_highlight_metrics
        self.log_metrics = log_metrics

        self.add_sparsity_loss = add_sparsity_loss
        self.sparsity_level = sparsity_level
        self.sparsity_coefficient = sparsity_coefficient
        self.add_continuity_loss = add_continuity_loss
        self.continuity_coefficient = continuity_coefficient
        self.drift_coefficient = drift_coefficient
        self.training_semantics = str(training_semantics).lower()
        self.manual_gradient_clip_val = manual_gradient_clip_val
        self.deterministic_budget_constraints = bool(deterministic_budget_constraints)
        self.per_example_budget = bool(per_example_budget)
        self.warmup_full_input = bool(warmup_full_input)
        self.mask_padding_constraints = bool(mask_padding_constraints)
        self.isolate_diagnostic_rng = bool(isolate_diagnostic_rng)
        self.collapse_recovery_budget = bool(collapse_recovery_budget)
        self.isolate_inactive_dropout = bool(isolate_inactive_dropout)
        self.diagnostic_seed = int(diagnostic_seed)
        self.classification_coefficient = classification_coefficient
        self.model_selection_rate_weight = model_selection_rate_weight
        self.model_selection_frag_weight = model_selection_frag_weight
        self.model_selection_min_f1 = model_selection_min_f1
        self.model_selection_min_rate = model_selection_min_rate
        self.model_selection_clf_penalty = model_selection_clf_penalty
        self.teacher_alignment_coefficient = teacher_alignment_coefficient
        if teacher_alignment_source not in {"online", "reference"}:
            raise ValueError(f"Unsupported teacher alignment source: {teacher_alignment_source}")
        if teacher_alignment_source == "reference" and teacher_alignment_mode != "grad_input":
            raise ValueError(
                "Reference teacher alignment requires `teacher_alignment_mode='grad_input'`."
            )
        self.teacher_alignment_source = teacher_alignment_source
        self.teacher_alignment_mode = teacher_alignment_mode
        self.teacher_alignment_schedule = teacher_alignment_schedule
        self.teacher_alignment_warmup_epochs = teacher_alignment_warmup_epochs
        self.teacher_alignment_confidence_threshold = float(teacher_alignment_confidence_threshold)
        self.teacher_alignment_require_correct = bool(teacher_alignment_require_correct)

        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self.enable_scheduling = (
            self._train_dataloader is not None and self._val_dataloader is not None
        )
        self.use_probe_diagnostics = bool(use_probe_diagnostics)
        self.use_reference_diagnostics = bool(use_reference_diagnostics)

        self.probe_train_batches = probe_train_batches
        self.probe_train_epochs = probe_train_epochs
        self.probe_lr = probe_lr
        self.probe_init_from_online = bool(probe_init_from_online)
        self.probe_calibration_coefficient = float(probe_calibration_coefficient)
        self.probe_calibration_temperature = float(probe_calibration_temperature)
        self.probe_calibration_warmup_epochs = max(int(probe_calibration_warmup_epochs), 0)
        self.probe_calibration_ramp_epochs = max(int(probe_calibration_ramp_epochs), 1)
        self.probe_calibration_min_gain = max(float(probe_calibration_min_gain), 0.0)
        self.ref_train_batches = ref_train_batches
        self.ref_train_epochs = ref_train_epochs
        self.ref_lr = ref_lr

        self.vrsr = VRSRScheduler(hp=vrsr_hparams)

        self.clf_loss = th.nn.CrossEntropyLoss()
        self.sparsity_loss = SparsityConstraint(
            sparsity_level,
            per_example=self.per_example_budget,
            mask_padding=self.mask_padding_constraints,
        )
        self.continuity_loss = ContinuityConstraint()
        self.reference_loss: Optional[float] = None
        self.reference_margin: Optional[float] = None
        self.reference_f1: Optional[float] = None
        self.reference_predictor: Optional[ProbePredictor] = None
        self.current_phase: str = self.vrsr.state.next_phase
        self.latest_probe_gap: float = 0.0
        self._probe_teacher = [None]
        self._probe_calibration_reliability: float = 0.0

        embedding = getattr(self.model, "embedding", None)
        if embedding is not None:
            self._embedding_requires_grad_base = [p.requires_grad for p in embedding.parameters()]
        else:
            self._embedding_requires_grad_base = None

        self.store_predictions = False
        self.val_predictions = []
        self.test_predictions = []

        self._prev_val_masks = {}
        self._curr_val_masks = {}
        self._val_probe_mask_cache = {}

        self.automatic_optimization = False

    def enable_storing_predictions(self):

        self.store_predictions = True

    def disable_storing_predictions(self):

        self.store_predictions = False

    def flush_predictions(self):

        self.val_predictions.clear()
        self.test_predictions.clear()

    def _set_requires_grad(
        self,
        module: th.nn.Module,
        requires_grad: bool,
    ):

        for p in module.parameters():
            p.requires_grad = requires_grad

    @contextmanager
    def _inactive_dropout_eval(self):

        if not self.isolate_inactive_dropout:
            yield
            return

        dropout_types = (
            th.nn.Dropout,
            th.nn.Dropout1d,
            th.nn.Dropout2d,
            th.nn.Dropout3d,
            th.nn.AlphaDropout,
            th.nn.FeatureAlphaDropout,
        )
        dropouts = [module for module in self.model.modules() if isinstance(module, dropout_types)]
        training_states = [bool(module.training) for module in dropouts]
        for module in dropouts:
            module.eval()
        try:
            yield
        finally:
            for module, was_training in zip(dropouts, training_states):
                module.train(was_training)

    def _apply_phase(
        self,
        phase: str,
    ):

        gen_modules = [
            getattr(self.model, "gen_encoder", None),
            getattr(self.model, "gen_classification_head", None),
            getattr(self.model, "gen_contextualizer", None),
            getattr(self.model, "mask_refiner", None),
            getattr(self.model, "layer_norm", None),
        ]
        gen_modules = [m for m in gen_modules if m is not None]

        pred_modules = [
            getattr(self.model, "cls_encoder", None),
            getattr(self.model, "classification_head", None),
        ]
        pred_modules = [m for m in pred_modules if m is not None]

        embedding = getattr(self.model, "embedding", None)

        if phase == "generator":
            if embedding is not None:
                for p in embedding.parameters():
                    p.requires_grad = False
            for m in gen_modules:
                self._set_requires_grad(m, True)
                m.train()
            for m in pred_modules:
                self._set_requires_grad(m, False)

                m.train()
        else:
            if embedding is not None and self._embedding_requires_grad_base is not None:
                for p, req in zip(embedding.parameters(), self._embedding_requires_grad_base):
                    p.requires_grad = req
            for m in gen_modules:
                self._set_requires_grad(m, False)
                m.eval()
            for m in pred_modules:
                self._set_requires_grad(m, True)
                m.train()

    def _build_probe_predictor(self) -> ProbePredictor:
        return self._build_probe_predictor_from_online(reset_parameters=True)

    def _build_probe_predictor_from_online(
        self,
        *,
        reset_parameters: bool,
    ) -> ProbePredictor:

        cls_encoder = copy.deepcopy(getattr(self.model, "cls_encoder"))
        classification_head = copy.deepcopy(getattr(self.model, "classification_head"))
        if reset_parameters:
            _reset_module_parameters(cls_encoder)
            _reset_module_parameters(classification_head)

        for p in cls_encoder.parameters():
            p.requires_grad = True
        for p in classification_head.parameters():
            p.requires_grad = True

        embedding = getattr(self.model, "embedding")
        dropout_p = float(getattr(getattr(self.model, "dropout", None), "p", 0.0))

        return ProbePredictor(
            embedding=embedding,
            cls_encoder=cls_encoder,
            classification_head=classification_head,
            dropout_p=dropout_p,
        )

    def _iter_limited(
        self,
        dataloader,
        max_batches: int,
    ):

        for batch_idx, batch in enumerate(dataloader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            yield batch

    def _get_cached_or_compute_highlight_mask(
        self,
        *,
        input_ids,
        attention_mask,
        sample_ids,
        cache: Optional[dict[int, th.Tensor]] = None,
    ):

        if cache is None:
            return self._deterministic_highlight_mask(input_ids, attention_mask, sample_ids)

        sample_ids_list = [int(sample_id) for sample_id in sample_ids.detach().cpu().tolist()]
        missing_sample = any(sample_id not in cache for sample_id in sample_ids_list)

        if missing_sample:
            with th.no_grad():
                computed_mask = self._deterministic_highlight_mask(
                    input_ids, attention_mask, sample_ids
                )
            computed_mask_cpu = computed_mask.detach().cpu()
            for idx, sample_id in enumerate(sample_ids_list):
                cache[sample_id] = computed_mask_cpu[idx]

        stacked_mask = th.stack([cache[sample_id] for sample_id in sample_ids_list], dim=0)
        return stacked_mask.to(device=input_ids.device, dtype=th.float32)

    def _train_and_eval_probe(
        self,
        *,
        full_mask: bool,
        train_batches: int,
        train_epochs: int,
        lr: float,
        keep_model: bool = False,
        reset_parameters: bool = True,
    ) -> tuple[float, float, Optional[ProbePredictor]]:

        if not self.isolate_diagnostic_rng:
            return self._train_and_eval_probe_unisolated(
                full_mask=full_mask,
                train_batches=train_batches,
                train_epochs=train_epochs,
                lr=lr,
                keep_model=keep_model,
                reset_parameters=reset_parameters,
            )

        devices = []
        if self.device.type == "cuda":
            device_index = self.device.index
            devices = [device_index if device_index is not None else th.cuda.current_device()]
        seed_offset = 0 if full_mask else 100000 + int(self.current_epoch)
        diagnostic_seed = self.diagnostic_seed + seed_offset
        with th.random.fork_rng(devices=devices, enabled=True):
            th.manual_seed(diagnostic_seed)
            if self.device.type == "cuda":
                th.cuda.manual_seed(diagnostic_seed)
            return self._train_and_eval_probe_unisolated(
                full_mask=full_mask,
                train_batches=train_batches,
                train_epochs=train_epochs,
                lr=lr,
                keep_model=keep_model,
                reset_parameters=reset_parameters,
            )

    def _train_and_eval_probe_unisolated(
        self,
        *,
        full_mask: bool,
        train_batches: int,
        train_epochs: int,
        lr: float,
        keep_model: bool = False,
        reset_parameters: bool = True,
    ) -> tuple[float, float, Optional[ProbePredictor]]:

        if self._train_dataloader is None or self._val_dataloader is None:
            return 0.0, 0.0, None

        with th.inference_mode(False):
            device = self.device

            embedding = getattr(self.model, "embedding", None)
            embedding_requires_grad = None
            if embedding is not None:
                embedding_requires_grad = [p.requires_grad for p in embedding.parameters()]
                for p in embedding.parameters():
                    p.requires_grad = False

            probe = self._build_probe_predictor_from_online(reset_parameters=reset_parameters).to(
                device
            )
            for p in probe.cls_encoder.parameters():
                p.requires_grad = True
            for p in probe.classification_head.parameters():
                p.requires_grad = True
            probe.train()

            probe_params = list(probe.cls_encoder.parameters()) + list(
                probe.classification_head.parameters()
            )
            optimizer = th.optim.Adam(probe_params, lr=lr)

            gen = getattr(self.model, "gen", None)
            gen_was_training = None
            if not full_mask and gen is not None:
                gen_was_training = bool(gen.training)
                gen.eval()

            train_mask_cache = {} if not full_mask else None
            val_mask_cache = (
                self._val_probe_mask_cache
                if (not full_mask and self._val_probe_mask_cache)
                else ({} if not full_mask else None)
            )

            with th.enable_grad():
                for _ in range(max(1, int(train_epochs))):
                    for batch in self._iter_limited(self._train_dataloader, int(train_batches)):
                        input_ids, attention_mask, sample_ids, y_true, _ = batch
                        input_ids = input_ids.to(device)
                        attention_mask = attention_mask.to(device)
                        sample_ids = sample_ids.to(device)
                        y_true = y_true.to(device)

                        if full_mask:
                            highlight_mask = attention_mask
                        else:
                            highlight_mask = self._get_cached_or_compute_highlight_mask(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                sample_ids=sample_ids,
                                cache=train_mask_cache,
                            )

                        logits = probe(input_ids, attention_mask, highlight_mask)
                        loss = self.clf_loss(logits, y_true)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

            if gen is not None and gen_was_training:
                gen.train()

            probe.eval()
            total_loss = 0.0
            total_margin = 0.0
            total_samples = 0
            confusion = th.zeros(
                (self.num_classes, self.num_classes),
                dtype=th.float64,
                device="cpu",
            )
            with th.no_grad():
                for batch in self._val_dataloader:
                    input_ids, attention_mask, sample_ids, y_true, _ = batch
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    sample_ids = sample_ids.to(device)
                    y_true = y_true.to(device)

                    if full_mask:
                        highlight_mask = attention_mask
                    else:
                        highlight_mask = self._get_cached_or_compute_highlight_mask(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            sample_ids=sample_ids,
                            cache=val_mask_cache,
                        )

                    logits = probe(input_ids, attention_mask, highlight_mask)
                    loss = self.clf_loss(logits, y_true)
                    margin = _mean_true_label_margin(logits, y_true)
                    bs = int(y_true.shape[0])
                    total_loss += float(loss.detach().cpu()) * bs
                    total_margin += float(margin.detach().cpu()) * bs
                    total_samples += bs
                    predictions = th.argmax(logits, dim=-1).detach().cpu()
                    labels = y_true.detach().to(device="cpu", dtype=th.long)
                    flat_indices = labels * self.num_classes + predictions
                    confusion += th.bincount(
                        flat_indices,
                        minlength=self.num_classes * self.num_classes,
                    ).reshape(self.num_classes, self.num_classes)

            if embedding is not None and embedding_requires_grad is not None:
                for p, req in zip(embedding.parameters(), embedding_requires_grad):
                    p.requires_grad = req

            total_samples = max(total_samples, 1)
            true_positives = confusion.diag()
            precision = true_positives / confusion.sum(dim=0).clamp_min(1.0)
            recall = true_positives / confusion.sum(dim=1).clamp_min(1.0)
            per_class_f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
            probe.diagnostic_macro_f1 = float(per_class_f1.mean())
            return (
                total_loss / total_samples,
                total_margin / total_samples,
                probe if keep_model else None,
            )

    def on_fit_start(self):

        if not self.enable_scheduling:
            return
        if not self.use_reference_diagnostics and self.teacher_alignment_source != "reference":
            return

        self.reference_loss, self.reference_margin, self.reference_predictor = (
            self._train_and_eval_probe(
                full_mask=True,
                train_batches=self.ref_train_batches,
                train_epochs=self.ref_train_epochs,
                lr=self.ref_lr,
                keep_model=True,
            )
        )
        if self.reference_predictor is not None:
            self.reference_f1 = float(getattr(self.reference_predictor, "diagnostic_macro_f1", 0.0))
            self.reference_predictor.eval()
            for parameter in self.reference_predictor.parameters():
                parameter.requires_grad = False

        if self.teacher_alignment_source == "reference":
            random_loss = math.log(max(int(self.num_classes), 2))
            reliable = (
                self.reference_loss is not None
                and math.isfinite(float(self.reference_loss))
                and float(self.reference_loss) < random_loss - 0.01
                and self.reference_f1 is not None
                and float(self.reference_f1) >= 0.55
            )
            if not reliable:
                raise RuntimeError(
                    "The full-input reference predictor is not reliable enough for "
                    "reference token guidance: "
                    f"loss={self.reference_loss}, macro_f1={self.reference_f1}. "
                    "Fix the reference fit before running the main scheduler."
                )

    def on_save_checkpoint(self, checkpoint):

        state_dict = checkpoint.get("state_dict")
        if not state_dict:
            return

        keys_to_drop = [key for key in state_dict.keys() if key.startswith("reference_predictor.")]
        for key in keys_to_drop:
            state_dict.pop(key, None)

    def on_train_epoch_start(self):

        if self.reference_loss is not None:
            self.log(
                "resr_ref_loss",
                float(self.reference_loss),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        if self.reference_margin is not None:
            self.log(
                "resr_ref_margin",
                float(self.reference_margin),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
        if self.reference_f1 is not None:
            self.log(
                "resr_ref_f1",
                float(self.reference_f1),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        self.current_phase = self.vrsr.state.next_phase
        self._apply_phase(self.current_phase)
        self.log(
            "resr_phase",
            1.0 if self.current_phase == "generator" else 0.0,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "resr_collapse_active_before_slice",
            1.0 if self.vrsr.state.collapse_active else 0.0,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "resr_ta_coef",
            float(self._current_teacher_alignment_coefficient()),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def on_train_epoch_end(self):

        if self.enable_scheduling:
            self.vrsr.on_slice_end(self.current_phase)

        self.log("resr_vr_g", float(self.vrsr.state.vr_g), on_step=False, on_epoch=True)
        self.log("resr_vr_p", float(self.vrsr.state.vr_p), on_step=False, on_epoch=True)
        self.log("resr_age_g", float(self.vrsr.state.age_g), on_step=False, on_epoch=True)
        self.log("resr_age_p", float(self.vrsr.state.age_p), on_step=False, on_epoch=True)

    def on_validation_epoch_start(self):

        if getattr(self.trainer, "sanity_checking", False):
            return
        self._curr_val_masks = {}
        self._val_probe_mask_cache = {}

    def configure_optimizers(self):

        gen_params = []
        pred_params = []

        if hasattr(self.model, "gen_encoder"):
            gen_params += list(self.model.gen_encoder.parameters())
        if hasattr(self.model, "gen_classification_head"):
            gen_params += list(self.model.gen_classification_head.parameters())
        if hasattr(self.model, "gen_contextualizer"):
            gen_params += list(self.model.gen_contextualizer.parameters())
        if hasattr(self.model, "mask_refiner") and self.model.mask_refiner is not None:
            gen_params += list(self.model.mask_refiner.parameters())
        if hasattr(self.model, "layer_norm"):
            gen_params += list(self.model.layer_norm.parameters())

        if hasattr(self.model, "embedding"):
            pred_params += list(self.model.embedding.parameters())
        if hasattr(self.model, "cls_encoder"):
            pred_params += list(self.model.cls_encoder.parameters())
        if hasattr(self.model, "classification_head"):
            pred_params += list(self.model.classification_head.parameters())

        opt_g = self.optimizer_class(gen_params, **self.optimizer_kwargs)
        opt_p = self.optimizer_class(pred_params, **self.optimizer_kwargs)
        return opt_g, opt_p

    def _compute_mask_stats(
        self,
        highlight_hat,
        attention_mask,
    ):

        attention_mask = attention_mask.to(dtype=th.float32)
        highlight_hat = highlight_hat.to(dtype=th.float32) * attention_mask

        if self.per_example_budget:
            valid_lengths = attention_mask.sum(dim=-1)
            valid_examples = valid_lengths > 0
            example_rates = highlight_hat.sum(dim=-1) / valid_lengths.clamp_min(1.0)
            rate = (
                example_rates[valid_examples].mean()
                if bool(valid_examples.any())
                else highlight_hat.sum() * 0.0
            )
        else:
            denom = th.clamp(attention_mask.sum(), min=1.0)
            rate = highlight_hat.sum() / denom

        if highlight_hat.shape[1] <= 1:
            frag = th.tensor(0.0, device=highlight_hat.device)
        else:
            valid_adj = attention_mask[:, 1:] * attention_mask[:, :-1]
            diffs = th.abs(highlight_hat[:, 1:] - highlight_hat[:, :-1]) * valid_adj
            frag = diffs.sum() / th.clamp(valid_adj.sum(), min=1.0)

        return rate, frag

    def _deterministic_highlight_mask(
        self,
        input_ids,
        attention_mask,
        sample_ids,
    ):
        _, highlight_logits = self.model.generator_forward(input_ids, attention_mask, sample_ids)
        return th.argmax(highlight_logits, dim=-1).to(dtype=th.float32)

    def _budget_constraint_mask(
        self,
        highlight_logits: th.Tensor,
        attention_mask: th.Tensor,
        stochastic_mask: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:

        return deterministic_budget_mask(
            highlight_logits=highlight_logits,
            attention_mask=attention_mask,
            stochastic_mask=stochastic_mask,
            temperature=float(getattr(self.model, "temperature", 1.0)),
            enabled=self.deterministic_budget_constraints,
            mask_padding=self.mask_padding_constraints,
        )

    @staticmethod
    def _mask_rates(mask: th.Tensor, attention_mask: th.Tensor) -> th.Tensor:
        return mask_rates(mask, attention_mask)

    @classmethod
    def _mean_mask_rate(cls, mask: th.Tensor, attention_mask: th.Tensor) -> th.Tensor:
        return cls._mask_rates(mask, attention_mask).mean()

    def _teacher_alignment_loss(
        self,
        highlight_logits,
        teacher_token_scores,
        attention_mask,
        sample_weights: Optional[th.Tensor] = None,
    ):
        valid_mask = attention_mask.to(dtype=th.bool)
        if highlight_logits.size(-1) > 1:
            generator_scores = highlight_logits[:, :, 1] - highlight_logits[:, :, 0]
        else:
            generator_scores = highlight_logits.squeeze(-1)

        mask_value = -1e9
        generator_scores = generator_scores.masked_fill(~valid_mask, mask_value)
        teacher_token_scores = teacher_token_scores.masked_fill(~valid_mask, mask_value)

        teacher_distribution = th.softmax(teacher_token_scores, dim=-1)
        generator_log_distribution = F.log_softmax(generator_scores, dim=-1)
        alignment = th.sum(
            teacher_distribution
            * (th.log(teacher_distribution.clamp_min(1e-8)) - generator_log_distribution),
            dim=-1,
        )

        valid_sequences = valid_mask.any(dim=-1)
        if valid_sequences.any():
            alignment = alignment[valid_sequences]
            if sample_weights is not None:
                weights = sample_weights.to(device=alignment.device, dtype=alignment.dtype)[
                    valid_sequences
                ]
                weight_sum = weights.sum()
                if bool((weight_sum > 0).detach().item()):
                    return (alignment * weights).sum() / weight_sum
                return alignment.sum() * 0.0
            return alignment.mean()

        return th.zeros((), device=highlight_logits.device, dtype=highlight_logits.dtype)

    def _teacher_alignment_gate(
        self,
        teacher_logits: th.Tensor,
        y_true: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        teacher_probs = F.softmax(teacher_logits.detach(), dim=-1)
        teacher_confidence, teacher_predictions = teacher_probs.max(dim=-1)
        weights = th.ones_like(teacher_confidence, dtype=teacher_confidence.dtype)

        confidence_threshold = float(self.teacher_alignment_confidence_threshold)
        if confidence_threshold > 0.0:
            weights = weights * (teacher_confidence >= confidence_threshold).to(dtype=weights.dtype)

        teacher_correct = teacher_predictions.eq(y_true.to(device=teacher_predictions.device))
        if self.teacher_alignment_require_correct:
            weights = weights * teacher_correct.to(dtype=weights.dtype)

        return weights, teacher_confidence, teacher_correct.to(dtype=weights.dtype)

    def _probe_calibration_scale(self) -> float:
        warmup_epochs = max(int(self.probe_calibration_warmup_epochs), 0)
        ramp_epochs = max(int(self.probe_calibration_ramp_epochs), 1)
        epoch = int(self.current_epoch)
        if epoch < warmup_epochs:
            return 0.0
        return min(float(epoch - warmup_epochs + 1) / float(ramp_epochs), 1.0)

    def _probe_calibration_active(self) -> bool:
        return (
            self.use_probe_diagnostics
            and self._probe_teacher[0] is not None
            and float(self.probe_calibration_coefficient) > 0.0
            and float(self._probe_calibration_reliability) > 0.0
            and self._probe_calibration_scale() > 0.0
        )

    def _current_teacher_alignment_coefficient(self) -> float:
        base_coefficient = float(self.teacher_alignment_coefficient)
        if base_coefficient <= 0.0:
            return 0.0

        schedule = str(self.teacher_alignment_schedule).lower()
        if schedule == "constant":
            return base_coefficient

        if schedule == "linear_warmup":
            warmup_epochs = max(int(self.teacher_alignment_warmup_epochs), 0)
            if warmup_epochs <= 0:
                return base_coefficient
            progress = min(float(self.current_epoch + 1) / float(warmup_epochs), 1.0)
            return base_coefficient * progress

        if schedule == "linear_decay":
            decay_epochs = max(int(self.teacher_alignment_warmup_epochs), 0)
            if decay_epochs <= 0:
                return 0.0
            progress = min(float(self.current_epoch) / float(decay_epochs), 1.0)
            return base_coefficient * (1.0 - progress)

        raise ValueError(
            f"Unsupported teacher alignment schedule: {self.teacher_alignment_schedule}"
        )

    def _generator_loss(
        self,
        input_ids,
        attention_mask,
        sample_ids,
        y_true,
    ):

        highlight_hat, highlight_logits = self.model.generator_forward(
            input_ids, attention_mask, sample_ids
        )
        with self._inactive_dropout_eval():
            logits = self.model.classifier_forward(
                input_ids, attention_mask, highlight_hat, sample_ids
            )
        constraint_mask, selection_prob = self._budget_constraint_mask(
            highlight_logits=highlight_logits,
            attention_mask=attention_mask,
            stochastic_mask=highlight_hat,
        )

        ce = self.clf_loss(logits, y_true)
        total = ce * self.classification_coefficient
        losses = {"CE": ce.detach()}

        if self.add_sparsity_loss:
            sp = self.sparsity_loss(highlight_hat=constraint_mask, attention_mask=attention_mask)
            total = total + sp * self.sparsity_coefficient
            losses["SP"] = sp.detach()

            recovery_budget = sp.sum() * 0.0
            if self.collapse_recovery_budget and self.vrsr.state.collapse_active:
                recovery_budget = collapse_rate_recovery_loss(
                    selection_prob=selection_prob,
                    attention_mask=attention_mask,
                    target_rate=self.vrsr.hp.r_star,
                )
                total = total + recovery_budget * self.sparsity_coefficient
            losses["RECOVERY_BUDGET"] = recovery_budget.detach()

        if self.add_continuity_loss:
            continuity_attention = attention_mask if self.mask_padding_constraints else None
            ct = self.continuity_loss(constraint_mask, attention_mask=continuity_attention)
            total = total + ct * self.continuity_coefficient
            losses["CT"] = ct.detach()

        losses["DET_RATE"] = self._mean_mask_rate(constraint_mask.detach(), attention_mask)
        losses["SOFT_RATE"] = self._mean_mask_rate(selection_prob.detach(), attention_mask)
        losses["COLLAPSE_ACTIVE"] = th.tensor(
            1.0 if self.vrsr.state.collapse_active else 0.0,
            device=logits.device,
            dtype=logits.dtype,
        )

        if self.drift_coefficient > 0.0 and self.vrsr.state.prev_val_rate is not None:
            rate, _ = self._compute_mask_stats(highlight_hat, attention_mask)
            dr = th.abs(rate - th.tensor(float(self.vrsr.state.prev_val_rate), device=rate.device))
            total = total + dr * float(self.drift_coefficient)
            losses["DR"] = dr.detach()

        current_teacher_alignment_coefficient = self._current_teacher_alignment_coefficient()
        if current_teacher_alignment_coefficient > 0.0:
            if self.teacher_alignment_source == "reference":
                teacher_model = self.reference_predictor
                teacher_forward = getattr(teacher_model, "teacher_guidance_forward", None)
            else:
                teacher_model = self.model
                teacher_forward = getattr(teacher_model, "teacher_guidance_forward", None)
            if not callable(teacher_forward):
                raise RuntimeError(
                    f"Teacher alignment source {self.teacher_alignment_source!r} "
                    "does not provide `teacher_guidance_forward(...)`."
                )
            if self.teacher_alignment_source == "reference":
                teacher_logits, teacher_token_scores = teacher_forward(
                    input_ids,
                    attention_mask,
                    target_labels=y_true,
                    method=self.teacher_alignment_mode,
                )
            elif self.teacher_alignment_mode == "grad_input":
                teacher_logits, teacher_token_scores = teacher_forward(
                    input_ids,
                    attention_mask,
                    sample_ids,
                    target_labels=y_true,
                    method=self.teacher_alignment_mode,
                )
            else:
                with th.no_grad():
                    teacher_logits, teacher_token_scores = teacher_forward(
                        input_ids,
                        attention_mask,
                        sample_ids,
                        target_labels=y_true,
                        method=self.teacher_alignment_mode,
                    )

            teacher_gate, teacher_confidence, teacher_correct = self._teacher_alignment_gate(
                teacher_logits=teacher_logits,
                y_true=y_true,
            )
            teacher_align = self._teacher_alignment_loss(
                highlight_logits=highlight_logits,
                teacher_token_scores=teacher_token_scores.detach(),
                attention_mask=attention_mask,
                sample_weights=teacher_gate,
            )

            total = total + teacher_align * float(current_teacher_alignment_coefficient)
            losses["TEA_KL"] = teacher_align.detach()
            losses["TEA_LOSS"] = teacher_align.detach()
            losses["TEA_COEF"] = th.tensor(
                float(current_teacher_alignment_coefficient),
                device=teacher_align.device,
            )
            losses["TEA_MODE"] = th.tensor(
                1.0 if self.teacher_alignment_mode == "grad_input" else 0.0,
                device=teacher_align.device,
            )
            losses["TEA_GATE"] = teacher_gate.detach().mean()
            losses["TEA_CONF"] = teacher_confidence.detach().mean()
            losses["TEA_ACC"] = teacher_correct.detach().mean()

        return total, logits, highlight_hat, losses

    def _predictor_loss(
        self,
        input_ids,
        attention_mask,
        sample_ids,
        y_true,
    ):

        warmup_full_input = self.warmup_full_input and int(self.current_epoch) < int(
            self.vrsr.hp.warmup_epochs
        )
        with th.no_grad():
            if warmup_full_input:
                highlight_hat = attention_mask.to(dtype=th.float32)
            else:
                with self._inactive_dropout_eval():
                    highlight_hat = self._deterministic_highlight_mask(
                        input_ids, attention_mask, sample_ids
                    )

        logits = self.model.classifier_forward(input_ids, attention_mask, highlight_hat, sample_ids)
        ce = self.clf_loss(logits, y_true)

        total = ce * self.classification_coefficient
        losses = {
            "CE": ce.detach(),
            "WARMUP_FULL_INPUT": th.tensor(
                1.0 if warmup_full_input else 0.0,
                device=logits.device,
                dtype=logits.dtype,
            ),
        }
        if self._probe_calibration_active():
            probe_teacher = self._probe_teacher[0]
            with th.no_grad():
                probe_logits = probe_teacher(input_ids, attention_mask, highlight_hat)
            probe_calibration_loss = _temperature_kl_from_logits(
                student_logits=logits,
                teacher_logits=probe_logits,
                temperature=self.probe_calibration_temperature,
            )
            scaled_probe_coefficient = (
                float(self.probe_calibration_coefficient)
                * float(self._probe_calibration_scale())
                * float(self._probe_calibration_reliability)
            )
            total = total + probe_calibration_loss * scaled_probe_coefficient
            losses["PCAL"] = probe_calibration_loss.detach()
            losses["PCAL_COEF"] = th.tensor(
                scaled_probe_coefficient,
                device=logits.device,
                dtype=logits.dtype,
            )
            losses["PCAL_REL"] = th.tensor(
                float(self._probe_calibration_reliability),
                device=logits.device,
                dtype=logits.dtype,
            )

        losses["PCAL_ACTIVE"] = th.tensor(
            1.0 if self._probe_calibration_active() else 0.0,
            device=logits.device,
            dtype=logits.dtype,
        )
        return total, logits, highlight_hat, losses

    def training_step(self, batch, batch_idx):

        input_ids, attention_mask, sample_ids, y_true, highlight_true = batch
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        sample_ids = sample_ids.to(self.device)
        y_true = y_true.to(self.device)

        opt_g, opt_p = self.optimizers()

        if self.current_phase == "generator":
            opt_g.zero_grad()
            loss, logits, highlight_hat, losses = self._generator_loss(
                input_ids, attention_mask, sample_ids, y_true
            )
            self.manual_backward(loss)
            if float(self.manual_gradient_clip_val) > 0.0:
                self.clip_gradients(
                    opt_g,
                    gradient_clip_val=float(self.manual_gradient_clip_val),
                    gradient_clip_algorithm="norm",
                )
            opt_g.step()
        else:
            opt_p.zero_grad()
            loss, logits, highlight_hat, losses = self._predictor_loss(
                input_ids, attention_mask, sample_ids, y_true
            )
            self.manual_backward(loss)
            if float(self.manual_gradient_clip_val) > 0.0:
                self.clip_gradients(
                    opt_p,
                    gradient_clip_val=float(self.manual_gradient_clip_val),
                    gradient_clip_algorithm="norm",
                )
            opt_p.step()

        self.log("train_loss", loss.detach(), on_step=False, on_epoch=True, prog_bar=True)
        for k, v in losses.items():
            self.log(f"train_{k}", v, on_step=False, on_epoch=True, prog_bar=False)

        rate, frag = self._compute_mask_stats(highlight_hat, attention_mask)
        self.log("train_hl_rate", rate.detach(), on_step=False, on_epoch=True)
        self.log("train_hl_frag", frag.detach(), on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):

        input_ids, attention_mask, sample_ids, y_true, highlight_true = batch
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        sample_ids = sample_ids.to(self.device)
        y_true = y_true.to(self.device)

        with th.no_grad():
            highlight_hat = self._deterministic_highlight_mask(
                input_ids, attention_mask, sample_ids
            )
            logits = self.model.classifier_forward(
                input_ids, attention_mask, highlight_hat, sample_ids
            )

        ce = self.clf_loss(logits, y_true)
        online_margin = _mean_true_label_margin(logits, y_true)
        self.log("val_loss", ce.detach(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_online_ce", ce.detach(), on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "val_online_margin",
            online_margin.detach(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        if self.val_classification_metrics is not None:
            y_pred = th.argmax(logits, dim=-1)
            self.val_classification_metrics.update(y_pred, y_true)
        else:
            y_pred = th.argmax(logits, dim=-1)

        if self.val_highlight_metrics is not None:
            self.val_highlight_metrics.update(highlight_hat, highlight_true)

        rate, frag = self._compute_mask_stats(highlight_hat, attention_mask)
        self.log("val_hl_rate", rate.detach(), on_step=False, on_epoch=True)
        self.log("val_hl_frag", frag.detach(), on_step=False, on_epoch=True)

        if not getattr(self.trainer, "sanity_checking", False):
            drift_values = []
            sample_ids_cpu = sample_ids.detach().cpu().tolist()
            attention_cpu = attention_mask.detach().cpu()
            highlight_cpu = highlight_hat.detach().cpu()
            for i, sample_id in enumerate(sample_ids_cpu):
                valid_len = int(attention_cpu[i].sum().item())
                curr_mask = (highlight_cpu[i, :valid_len] > 0.5).to(dtype=th.uint8)
                prev_mask = self._prev_val_masks.get(int(sample_id))
                if (
                    prev_mask is not None
                    and int(prev_mask.numel()) == int(curr_mask.numel())
                    and valid_len > 0
                ):
                    drift_values.append(
                        th.abs(
                            curr_mask.to(dtype=th.float32) - prev_mask.to(dtype=th.float32)
                        ).mean()
                    )
                self._curr_val_masks[int(sample_id)] = curr_mask
                self._val_probe_mask_cache[int(sample_id)] = highlight_cpu[i]

            if len(drift_values):
                batch_drift = th.stack(drift_values).mean()
            else:
                batch_drift = th.tensor(0.0)
            self.log("val_hl_drift", batch_drift, on_step=False, on_epoch=True)

        if self.store_predictions:
            self.val_predictions.append(
                {
                    "y_true": y_true.detach().cpu(),
                    "y_hat": y_pred.detach().cpu(),
                    "hl_true": highlight_true.detach().cpu(),
                    "hl_hat": highlight_hat.detach().cpu(),
                    "input_ids": input_ids.detach().cpu(),
                    "sample_ids": sample_ids.detach().cpu(),
                }
            )

        return ce

    def on_validation_epoch_end(self):

        val_classification_values = None
        val_highlight_values = None

        if self.val_classification_metrics is not None:
            val_classification_values = self.val_classification_metrics.compute()
            for key, value in val_classification_values.items():
                self.log(f"val_{key}", value, prog_bar=self.log_metrics)
            self.val_classification_metrics.reset()

        if self.val_highlight_metrics is not None:
            val_highlight_values = self.val_highlight_metrics.compute()
            for key, value in val_highlight_values.items():
                self.log(f"val_{key}", value, prog_bar=self.log_metrics)
            self.val_highlight_metrics.reset()

        val_rate = self.trainer.callback_metrics.get("val_hl_rate")
        val_frag = self.trainer.callback_metrics.get("val_hl_frag")
        val_loss = self.trainer.callback_metrics.get("val_loss")
        val_hl_pos_f1 = None
        val_f1 = None
        if val_highlight_values is not None:
            val_hl_pos_f1 = val_highlight_values.get("hl_pos_f1")
        if val_classification_values is not None:
            val_f1 = val_classification_values.get("f1")

        if val_loss is not None and val_rate is not None:
            structural_loss, structural_feasible = structural_checkpoint_loss(
                validation_loss=val_loss.detach(),
                validation_rate=val_rate.detach(),
                target_rate=float(self.vrsr.hp.r_star),
            )
            self.log(
                "val_structural_loss",
                structural_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                "val_structural_feasible",
                structural_feasible,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

        if val_hl_pos_f1 is not None and val_rate is not None and val_frag is not None:
            rate_target = (
                float(self.vrsr.hp.r_star) if self.enable_scheduling else float(self.sparsity_level)
            )
            rate_gap = th.abs(val_rate.detach().to(dtype=th.float32) - rate_target)
            clf_penalty = th.tensor(0.0, device=rate_gap.device)
            if val_f1 is not None and float(self.model_selection_clf_penalty) > 0.0:
                clf_gap = th.relu(
                    th.tensor(float(self.model_selection_min_f1), device=rate_gap.device)
                    - val_f1.detach().to(dtype=th.float32)
                )
                clf_penalty = float(self.model_selection_clf_penalty) * clf_gap
            rate_floor_penalty = th.tensor(0.0, device=rate_gap.device)
            if float(self.model_selection_min_rate) > 0.0:
                rate_floor_gap = th.relu(
                    th.tensor(float(self.model_selection_min_rate), device=rate_gap.device)
                    - val_rate.detach().to(dtype=th.float32)
                )
                if bool((rate_floor_gap > 0.0).detach().cpu().item()):
                    rate_floor_penalty = 1.0 + 10.0 * rate_floor_gap
            select_score = (
                val_hl_pos_f1.detach().to(dtype=th.float32)
                - float(self.model_selection_rate_weight) * rate_gap
                - float(self.model_selection_frag_weight) * val_frag.detach().to(dtype=th.float32)
                - clf_penalty
                - rate_floor_penalty
            )
            self.log(
                "val_select_score",
                select_score,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

        if getattr(self.trainer, "sanity_checking", False):
            return

        if not self.enable_scheduling:
            self._prev_val_masks = self._curr_val_masks
            return

        l_online = self.trainer.callback_metrics.get("val_online_ce")
        m_online = self.trainer.callback_metrics.get("val_online_margin")
        l_online_f = float(l_online.detach().cpu()) if l_online is not None else 0.0
        m_online_f = float(m_online.detach().cpu()) if m_online is not None else 0.0
        val_rate_f = float(val_rate.detach().cpu()) if val_rate is not None else 0.0
        val_frag_f = float(val_frag.detach().cpu()) if val_frag is not None else 0.0
        val_drift = self.trainer.callback_metrics.get("val_hl_drift")
        val_drift_f = float(val_drift.detach().cpu()) if val_drift is not None else 0.0
        if self.use_reference_diagnostics and self.reference_loss is not None:
            l_ref_f = float(self.reference_loss)
            m_ref_f = (
                float(self.reference_margin)
                if self.reference_margin is not None
                else float(m_online_f)
            )
        else:
            l_ref_f = None
            m_ref_f = None

        if self.use_probe_diagnostics:
            l_probe_f, m_probe_f, probe_teacher = self._train_and_eval_probe(
                full_mask=False,
                train_batches=self.probe_train_batches,
                train_epochs=self.probe_train_epochs,
                lr=self.probe_lr,
                keep_model=True,
                reset_parameters=not self.probe_init_from_online,
            )
            if probe_teacher is not None:
                probe_teacher.eval()
                for p in probe_teacher.parameters():
                    p.requires_grad = False
            self._probe_teacher[0] = probe_teacher
            probe_gain = max(
                0.0,
                float(l_online_f) - float(l_probe_f) - float(self.probe_calibration_min_gain),
            )
            self._probe_calibration_reliability = min(
                1.0,
                probe_gain / max(abs(float(l_online_f)), 1e-6),
            )
            l_probe_for_predictor = float(l_probe_f)
            self.log("resr_probe_disabled", 0.0, on_step=False, on_epoch=True, prog_bar=False)
        else:
            l_probe_f = float(l_ref_f) if l_ref_f is not None else float(l_online_f)
            m_probe_f = float(m_ref_f) if m_ref_f is not None else float(m_online_f)
            l_probe_for_predictor = float(l_online_f)
            self._probe_teacher[0] = None
            self._probe_calibration_reliability = 0.0
            self.log("resr_probe_disabled", 1.0, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "resr_probe_loss",
            float(l_probe_f),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "resr_probe_margin",
            float(m_probe_f),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "resr_probe_calibration_reliability",
            float(self._probe_calibration_reliability),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        if l_ref_f is None:
            l_ref_f = float(l_probe_f)
            m_ref_f = float(m_probe_f)
        self.latest_probe_gap = max(0.0, float(l_probe_f) - (l_ref_f + float(self.vrsr.hp.epsilon)))
        self.log(
            "resr_probe_gap",
            float(self.latest_probe_gap),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "resr_predictor_gap",
            max(0.0, float(l_online_f) - float(l_probe_for_predictor)),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        decision = self.vrsr.decide(
            l_online=l_online_f,
            l_probe=float(l_probe_f),
            l_ref=l_ref_f,
            l_probe_predictor=float(l_probe_for_predictor),
            val_rate=val_rate_f,
            val_frag=val_frag_f,
            val_drift=val_drift_f,
        )

        self.log("resr_Dg", float(decision.d_g), on_step=False, on_epoch=True)
        self.log("resr_Dp", float(decision.d_p), on_step=False, on_epoch=True)
        self.log(
            "resr_probe_gap_effective",
            float(decision.generator_gap),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "resr_predictor_gap_effective",
            float(decision.predictor_gap),
            on_step=False,
            on_epoch=True,
        )
        self.log("resr_wg", float(decision.w_g), on_step=False, on_epoch=True)
        self.log("resr_wp", float(decision.w_p), on_step=False, on_epoch=True)
        self.log(
            "resr_next_phase",
            1.0 if decision.next_phase == "generator" else 0.0,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "resr_collapse_active",
            1.0 if decision.collapse_active else 0.0,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "resr_collapse_events",
            float(decision.collapse_events),
            on_step=False,
            on_epoch=True,
        )
        if decision.reason == "warmup":
            reason_code = 1.0
        elif decision.reason in {"mask_collapse", "forced:generator"}:
            reason_code = 2.0
        elif decision.reason == "inherit:predictor":
            reason_code = 3.0
        elif decision.reason == "inherit:generator":
            reason_code = 4.0
        else:
            reason_code = 0.0
        self.log("resr_decision_reason", reason_code, on_step=False, on_epoch=True)

        self._prev_val_masks = self._curr_val_masks
        self._val_probe_mask_cache = {}

    def test_step(self, batch, batch_idx):

        input_ids, attention_mask, sample_ids, y_true, highlight_true = batch
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        sample_ids = sample_ids.to(self.device)
        y_true = y_true.to(self.device)

        with th.no_grad():
            highlight_hat = self._deterministic_highlight_mask(
                input_ids, attention_mask, sample_ids
            )
            logits = self.model.classifier_forward(
                input_ids, attention_mask, highlight_hat, sample_ids
            )

        ce = self.clf_loss(logits, y_true)
        self.log("test_loss", ce.detach(), on_step=False, on_epoch=True, prog_bar=True)

        if self.test_classification_metrics is not None:
            y_pred = th.argmax(logits, dim=-1)
            self.test_classification_metrics.update(y_pred, y_true)
        else:
            y_pred = th.argmax(logits, dim=-1)

        if self.test_highlight_metrics is not None:
            self.test_highlight_metrics.update(highlight_hat, highlight_true)

        if self.store_predictions:
            self.test_predictions.append(
                {
                    "y_true": y_true.detach().cpu(),
                    "y_hat": y_pred.detach().cpu(),
                    "hl_true": highlight_true.detach().cpu(),
                    "hl_hat": highlight_hat.detach().cpu(),
                    "input_ids": input_ids.detach().cpu(),
                    "sample_ids": sample_ids.detach().cpu(),
                }
            )

        return ce

    def on_test_epoch_end(self):

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
