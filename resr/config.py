from pathlib import Path
import sys

import torch as th


RESR_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RESR_ROOT.parent
BASELINES_ROOT = PROJECT_ROOT.joinpath("baselines")
if str(BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINES_ROOT))

from configurations.base import BaseConfig, ConfigKey


DEFAULT_MAIN_SEEDS = [2023, 1337, 12345, 31415, 42]


TRAINING_SEMANTICS_PROFILES = {
    "paper": {
        "manual_gradient_clip_val": 0.0,
        "deterministic_budget_constraints": False,
        "per_example_budget": False,
        "warmup_full_input": False,
        "mask_padding_constraints": False,
        "isolate_diagnostic_rng": False,
        "collapse_recovery_budget": False,
        "isolate_inactive_dropout": False,
    },
}


def get_training_semantics_defaults(training_semantics: str) -> dict:

    name = str(training_semantics).lower()
    if name not in TRAINING_SEMANTICS_PROFILES:
        choices = ", ".join(sorted(TRAINING_SEMANTICS_PROFILES))
        raise ValueError(
            f"Unsupported training semantics: {training_semantics}. Choose from: {choices}"
        )
    return dict(TRAINING_SEMANTICS_PROFILES[name])


class ReSRConfig(BaseConfig):
    configs = {
        ConfigKey(dataset="toy", tags={"spp", "resr"}): "toy",
        ConfigKey(dataset="hatexplain", tags={"spp", "resr"}): "hatexplain",
    }

    def __init__(
        self,
        embedding_dim,
        hidden_size,
        classification_head,
        selection_head,
        dropout_rate=0.1,
        temperature=1.0,
        embedding_name=None,
        scheduler_policy: str = "vrsr",
        r_star: float = 0.15,
        epsilon: float = 0.0,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        delta: float = 1.0,
        eta: float = 0.0,
        mu: float = 1.0,
        nu: float = 0.0,
        q: float = 1.0,
        warmup_epochs: int = 0,
        inherit_gap_threshold: float = 0.0,
        inherit_patience: int = 0,
        inherit_slices: int = 2,
        probe_gap_threshold: float = 0.0,
        predictor_gap_threshold: float = 0.0,
        gap_ema_decay: float = 0.0,
        collapse_min_rate: float = 0.0,
        collapse_max_rate: float = 1.0,
        recovery_slices: int = 2,
        use_generator_scheduler_guidance: bool = True,
        use_predictor_scheduler_guidance: bool = True,
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
        drift_coefficient: float = 0.0,
        training_semantics: str = "stabilized",
        manual_gradient_clip_val=None,
        deterministic_budget_constraints=None,
        per_example_budget=None,
        warmup_full_input=None,
        mask_padding_constraints=None,
        isolate_diagnostic_rng=None,
        collapse_recovery_budget=None,
        isolate_inactive_dropout=None,
        diagnostic_seed: int = 15000,
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
        early_stopping_patience: int = 120,
        checkpoint_monitor: str = "val_select_score",
        checkpoint_mode: str = "max",
        has_validation_highlights: bool = True,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.classification_head = classification_head
        self.selection_head = selection_head
        self.dropout_rate = dropout_rate
        self.temperature = temperature
        self.embedding_name = embedding_name

        if scheduler_policy not in {"vrsr", "fixed_alternating", "loss_gap"}:
            raise ValueError(f"Unsupported scheduler policy: {scheduler_policy}")
        self.scheduler_policy = scheduler_policy
        self.r_star = r_star
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.eta = eta
        self.mu = mu
        self.nu = nu
        self.q = q
        self.warmup_epochs = warmup_epochs
        self.inherit_gap_threshold = inherit_gap_threshold
        self.inherit_patience = inherit_patience
        self.inherit_slices = inherit_slices
        self.probe_gap_threshold = probe_gap_threshold
        self.predictor_gap_threshold = predictor_gap_threshold
        self.gap_ema_decay = gap_ema_decay
        self.collapse_min_rate = collapse_min_rate
        self.collapse_max_rate = collapse_max_rate
        self.recovery_slices = recovery_slices
        self.use_generator_scheduler_guidance = use_generator_scheduler_guidance
        self.use_predictor_scheduler_guidance = use_predictor_scheduler_guidance

        self.probe_train_batches = probe_train_batches
        self.probe_train_epochs = probe_train_epochs
        self.probe_lr = probe_lr
        self.probe_init_from_online = probe_init_from_online
        self.probe_calibration_coefficient = probe_calibration_coefficient
        self.probe_calibration_temperature = probe_calibration_temperature
        self.probe_calibration_warmup_epochs = probe_calibration_warmup_epochs
        self.probe_calibration_ramp_epochs = probe_calibration_ramp_epochs
        self.probe_calibration_min_gain = probe_calibration_min_gain
        self.ref_train_batches = ref_train_batches
        self.ref_train_epochs = ref_train_epochs
        self.ref_lr = ref_lr

        semantics_name = str(training_semantics).lower()
        semantics_defaults = get_training_semantics_defaults(semantics_name)

        def semantic_value(name, explicit_value):
            if explicit_value is None:
                return semantics_defaults[name]
            return explicit_value

        self.drift_coefficient = drift_coefficient
        self.training_semantics = semantics_name
        self.manual_gradient_clip_val = float(
            semantic_value("manual_gradient_clip_val", manual_gradient_clip_val)
        )
        self.deterministic_budget_constraints = bool(
            semantic_value("deterministic_budget_constraints", deterministic_budget_constraints)
        )
        self.per_example_budget = bool(semantic_value("per_example_budget", per_example_budget))
        self.warmup_full_input = bool(semantic_value("warmup_full_input", warmup_full_input))
        self.mask_padding_constraints = bool(
            semantic_value("mask_padding_constraints", mask_padding_constraints)
        )
        self.isolate_diagnostic_rng = bool(
            semantic_value("isolate_diagnostic_rng", isolate_diagnostic_rng)
        )
        self.collapse_recovery_budget = bool(
            semantic_value("collapse_recovery_budget", collapse_recovery_budget)
        )
        self.isolate_inactive_dropout = bool(
            semantic_value("isolate_inactive_dropout", isolate_inactive_dropout)
        )
        self.diagnostic_seed = int(diagnostic_seed)
        self.model_selection_rate_weight = model_selection_rate_weight
        self.model_selection_frag_weight = model_selection_frag_weight
        self.model_selection_min_f1 = model_selection_min_f1
        self.model_selection_min_rate = model_selection_min_rate
        self.model_selection_clf_penalty = model_selection_clf_penalty
        self.teacher_alignment_coefficient = teacher_alignment_coefficient
        if teacher_alignment_source not in {"online", "reference"}:
            raise ValueError(f"Unsupported teacher alignment source: {teacher_alignment_source}")
        self.teacher_alignment_source = teacher_alignment_source
        self.teacher_alignment_mode = teacher_alignment_mode
        self.teacher_alignment_schedule = teacher_alignment_schedule
        self.teacher_alignment_warmup_epochs = teacher_alignment_warmup_epochs
        self.teacher_alignment_confidence_threshold = teacher_alignment_confidence_threshold
        self.teacher_alignment_require_correct = teacher_alignment_require_correct
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_monitor = checkpoint_monitor
        self.checkpoint_mode = checkpoint_mode
        self.has_validation_highlights = has_validation_highlights

    @classmethod
    def toy(cls):

        return cls(
            seeds=DEFAULT_MAIN_SEEDS,
            optimizer_class=th.optim.Adam,
            optimizer_kwargs={"lr": 1e-03},
            batch_size=64,
            embedding_dim=25,
            hidden_size=8,
            classification_head=lambda: th.nn.Sequential(th.nn.Linear(16, 3)),
            selection_head=lambda: th.nn.Sequential(th.nn.Linear(16, 2)),
            num_classes=3,
            dropout_rate=0.0,
            freeze_embeddings=True,
            add_sparsity_loss=True,
            sparsity_coefficient=1.5,
            sparsity_level=0.15,
            add_continuity_loss=True,
            continuity_coefficient=2.0,
            classification_coefficient=1.0,
            r_star=0.15,
            epsilon=0.0,
            alpha=1.0,
            beta=1.0,
            gamma=1.0,
            delta=1.0,
            eta=0.1,
            mu=1.0,
            nu=0.1,
            q=1.0,
            warmup_epochs=1,
            inherit_gap_threshold=0.25,
            inherit_patience=3,
            inherit_slices=2,
            probe_gap_threshold=0.02,
            predictor_gap_threshold=0.02,
            gap_ema_decay=0.6,
            collapse_min_rate=0.05,
            collapse_max_rate=0.98,
            recovery_slices=5,
            probe_train_batches=50,
            probe_train_epochs=3,
            probe_lr=1e-3,
            probe_init_from_online=True,
            probe_calibration_coefficient=0.08,
            probe_calibration_temperature=2.0,
            probe_calibration_warmup_epochs=5,
            probe_calibration_ramp_epochs=5,
            probe_calibration_min_gain=0.005,
            ref_train_batches=100,
            ref_train_epochs=3,
            ref_lr=1e-3,
            drift_coefficient=0.0,
            training_semantics="paper",
            diagnostic_seed=15000,
            model_selection_rate_weight=4.0,
            model_selection_frag_weight=1.0,
            model_selection_min_f1=0.97,
            model_selection_min_rate=0.05,
            model_selection_clf_penalty=5.0,
            teacher_alignment_coefficient=0.02,
            teacher_alignment_mode="margin",
            teacher_alignment_schedule="linear_decay",
            teacher_alignment_warmup_epochs=30,
            teacher_alignment_confidence_threshold=0.60,
            teacher_alignment_require_correct=False,
            early_stopping_patience=120,
        )

    @classmethod
    def hatexplain(cls):

        return cls(
            seeds=DEFAULT_MAIN_SEEDS,
            optimizer_class=th.optim.Adam,
            optimizer_kwargs={"lr": 1e-03},
            batch_size=64,
            embedding_dim=25,
            hidden_size=16,
            classification_head=lambda: th.nn.Sequential(th.nn.Linear(32, 2)),
            selection_head=lambda: th.nn.Sequential(th.nn.Linear(32, 2)),
            num_classes=2,
            dropout_rate=0.0,
            freeze_embeddings=True,
            use_pretrained_only=True,
            embedding_name="twitter.27B",
            add_sparsity_loss=True,
            sparsity_coefficient=1.0,
            sparsity_level=0.22,
            add_continuity_loss=False,
            continuity_coefficient=1.0,
            classification_coefficient=1.0,
            r_star=0.22,
            epsilon=0.0,
            alpha=1.0,
            beta=1.0,
            gamma=1.0,
            delta=1.0,
            eta=0.1,
            mu=1.0,
            nu=0.1,
            q=1.0,
            warmup_epochs=1,
            inherit_gap_threshold=0.25,
            inherit_patience=3,
            inherit_slices=2,
            probe_gap_threshold=0.03,
            predictor_gap_threshold=0.03,
            gap_ema_decay=0.6,
            collapse_min_rate=0.02,
            collapse_max_rate=0.98,
            recovery_slices=2,
            probe_train_batches=10,
            probe_train_epochs=1,
            probe_lr=1e-3,
            probe_init_from_online=True,
            probe_calibration_coefficient=0.08,
            probe_calibration_temperature=2.0,
            probe_calibration_warmup_epochs=5,
            probe_calibration_ramp_epochs=5,
            probe_calibration_min_gain=0.005,
            ref_train_batches=50,
            ref_train_epochs=3,
            ref_lr=1e-3,
            drift_coefficient=0.0,
            training_semantics="paper",
            diagnostic_seed=15000,
            model_selection_rate_weight=1.0,
            model_selection_frag_weight=0.25,
            model_selection_min_f1=0.68,
            model_selection_clf_penalty=2.0,
            teacher_alignment_coefficient=0.02,
            teacher_alignment_mode="margin",
            teacher_alignment_schedule="linear_decay",
            teacher_alignment_warmup_epochs=30,
            teacher_alignment_confidence_threshold=0.60,
            teacher_alignment_require_correct=False,
            early_stopping_patience=120,
        )
