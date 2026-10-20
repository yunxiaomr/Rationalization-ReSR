import csv
import hashlib
import json
import logging
import platform
import shutil
import sys
from pathlib import Path

import pytorch_lightning as L
import numpy as np
import torch as th
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification.f_beta import F1Score

RESR_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RESR_ROOT.parent
BASELINES_ROOT = PROJECT_ROOT.joinpath("baselines")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINES_ROOT))

from components.callbacks import PycharmProgressBar
from components.data_loader import build_loader_for_dataset
from components.metrics import BinaryHighlightF1Score, SelectionRate, SelectionSize
from resr.components.resr import ReSRHlModel
from components.processing import (
    GloVeEmbedderCollator,
    HighlightDataset,
    HighlightExtractor,
    OneHotEmbedderCollator,
)
from configurations.base import ConfigKey
from resr.config import ReSRConfig
from resr.modeling.stability import target_relative_rate_bounds
from resr.modeling.vrsr import VRSRHyperparams
from modeling.spp import MCD
from runnables.benchmark_cli import parse_runner_args, run_path


def parse_resr_args(description: str):
    return parse_runner_args(description)


def run_resr(dataset: str, args) -> None:
    logging.basicConfig(level=logging.INFO)

    dataset = dataset.lower()
    if dataset not in {"hatexplain", "toy"}:
        raise ValueError(f"Unsupported ReSR dataset: {dataset}")

    config = ReSRConfig.from_config(key=ConfigKey(dataset=dataset, tags={"spp", "resr"}))
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    logging.info("Running ReSR on %s", dataset)

    resolved = _resolve_run_config(config, args)
    dataset_root = RESR_ROOT.joinpath("results", dataset)
    save_path = run_path(dataset_root / "resr_main", args)
    save_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_path.joinpath("checkpoints")
    ckpt_path.mkdir(parents=True, exist_ok=True)

    _log_run_config(resolved, config)

    seed_everything(seed=15000, workers=True)
    train_data, val_data, test_data, collator, extractor = _build_data(
        dataset, config, num_workers=args.num_workers
    )
    vrsr_hparams = _build_vrsr_hparams(config, resolved)
    selected_seeds = list(args.seeds if args.seeds is not None else config.seeds)
    if len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError(f"Duplicate seeds are not allowed: {selected_seeds}")
    if args.force_retrain:
        _clear_stale_run_outputs(save_path, selected_seeds)

    _write_json_atomic(
        save_path / "run_config.json",
        {
            "dataset": dataset,
            "method": "ReSR",
            "selected_seeds": [int(seed) for seed in selected_seeds],
            **resolved,
        },
    )
    _write_reproducibility_manifest(
        save_path=save_path,
        dataset=dataset,
        selected_seeds=selected_seeds,
        loaders={"train": train_data, "validation": val_data, "test": test_data},
        resolved=resolved,
    )

    metrics = {}
    seed_records = []
    trainer_args = {
        "accelerator": "auto",
        "devices": 1,
        "accumulate_grad_batches": 1,
        "max_epochs": args.max_epochs if args.max_epochs is not None else config.max_epochs,
    }
    for trainer_key, arg_name in [
        ("limit_train_batches", "limit_train_batches"),
        ("limit_val_batches", "limit_val_batches"),
        ("limit_test_batches", "limit_test_batches"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            trainer_args[trainer_key] = _lightning_limit_value(value)

    for seed in selected_seeds:
        seed_everything(seed=seed, workers=True)
        seed_ckpt_path = ckpt_path.joinpath(f"seed={seed}")
        seed_ckpt_path.mkdir(parents=True, exist_ok=True)
        seed_log_path = save_path.joinpath("training_logs", f"seed={seed}")

        existing_ckpts = _list_checkpoints(seed_ckpt_path)
        if args.force_retrain:
            _clear_checkpoints(seed_ckpt_path)
            existing_ckpts = []
            if seed_log_path.exists():
                shutil.rmtree(seed_log_path)

        model = _build_resr_model(
            config=config,
            collator=collator,
            vrsr_hparams=vrsr_hparams,
            train_data=train_data,
            val_data=val_data,
            resolved=resolved,
        )

        should_train = args.force_retrain or not existing_ckpts
        checkpoint_monitor = resolved["checkpoint_monitor"]
        checkpoint_mode = resolved["checkpoint_mode"]
        checkpoint_callback = ModelCheckpoint(
            monitor=checkpoint_monitor,
            mode=checkpoint_mode,
            dirpath=seed_ckpt_path,
        )
        csv_logger = CSVLogger(
            save_dir=save_path.joinpath("training_logs").as_posix(),
            name=f"seed={seed}",
            version="run",
        )
        trainer = L.Trainer(
            **trainer_args,
            callbacks=[
                EarlyStopping(
                    monitor=checkpoint_monitor,
                    mode=checkpoint_mode,
                    patience=resolved["early_stopping_patience"],
                ),
                checkpoint_callback,
                PycharmProgressBar(),
            ],
            deterministic=True,
            logger=csv_logger,
        )

        if should_train:
            trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)
            if checkpoint_callback.best_model_path:
                ckpt_filepath = Path(checkpoint_callback.best_model_path)
            else:
                existing_ckpts = _list_checkpoints(seed_ckpt_path)
                if not existing_ckpts:
                    raise FileNotFoundError(
                        f"No checkpoint was produced for seed {seed} in {seed_ckpt_path}."
                    )
                ckpt_filepath = existing_ckpts[0]
        else:
            logging.info("Found existing checkpoint! Skipping model training...")
            ckpt_filepath = existing_ckpts[0]

        logging.info("Using checkpoint for seed %s: %s", seed, ckpt_filepath)
        model = _load_resr_model(
            ckpt_filepath=ckpt_filepath,
            config=config,
            collator=collator,
            vrsr_hparams=vrsr_hparams,
            resolved=resolved,
        )

        seed_everything(seed=seed, workers=True)
        val_metrics = trainer.validate(model=model, dataloaders=val_data)[0]
        logging.info("Validation metrics: %s", val_metrics)

        model.enable_storing_predictions()
        test_metrics = trainer.test(model=model, dataloaders=test_data)[0]
        test_predictions = model.test_predictions
        logging.info("Test metrics: %s", test_metrics)

        for metric_name, metric_value in val_metrics.items():
            metrics.setdefault("validation", {}).setdefault(metric_name, []).append(metric_value)
        for metric_name, metric_value in test_metrics.items():
            metrics.setdefault("test", {}).setdefault(metric_name, []).append(metric_value)

        seed_records.append(
            _make_seed_record(
                seed=seed,
                checkpoint=ckpt_filepath,
                checkpoint_monitor=checkpoint_monitor,
                validation_metrics=val_metrics,
                test_metrics=test_metrics,
            )
        )
        _write_seed_records(seed_records, save_path)

        extractor(
            predictions=test_predictions,
            vocabulary=collator.vocabulary,
            save_path=save_path,
            seed=seed,
        )
        model.flush_predictions()
        model.disable_storing_predictions()

    _summarize_and_save_metrics(metrics, selected_seeds, seed_records, save_path)


def _resolve_run_config(config: ReSRConfig, args) -> dict:
    fields = (
        "scheduler_policy",
        "r_star",
        "collapse_min_rate",
        "collapse_max_rate",
        "training_semantics",
        "teacher_alignment_coefficient",
        "teacher_alignment_source",
        "teacher_alignment_mode",
        "teacher_alignment_schedule",
        "teacher_alignment_warmup_epochs",
        "teacher_alignment_confidence_threshold",
        "teacher_alignment_require_correct",
        "model_selection_rate_weight",
        "model_selection_frag_weight",
        "model_selection_min_f1",
        "model_selection_min_rate",
        "model_selection_clf_penalty",
        "early_stopping_patience",
        "probe_train_batches",
        "probe_train_epochs",
        "probe_lr",
        "probe_init_from_online",
        "probe_calibration_coefficient",
        "probe_calibration_temperature",
        "probe_calibration_warmup_epochs",
        "probe_calibration_ramp_epochs",
        "probe_calibration_min_gain",
        "probe_gap_threshold",
        "predictor_gap_threshold",
        "gap_ema_decay",
        "use_generator_scheduler_guidance",
        "use_predictor_scheduler_guidance",
        "deterministic_budget_constraints",
        "per_example_budget",
        "warmup_full_input",
        "mask_padding_constraints",
        "isolate_diagnostic_rng",
        "collapse_recovery_budget",
        "isolate_inactive_dropout",
        "diagnostic_seed",
        "manual_gradient_clip_val",
        "checkpoint_monitor",
        "checkpoint_mode",
    )
    resolved = {name: getattr(config, name) for name in fields}
    lower, upper = target_relative_rate_bounds(config.r_star, lower_factor=0.5, upper_factor=2.0)
    resolved.update(
        structural_min_rate=lower,
        structural_max_rate=upper,
        runtime={
            "max_epochs": args.max_epochs if args.max_epochs is not None else config.max_epochs,
            "batch_size": args.batch_size if args.batch_size is not None else config.batch_size,
            "limit_train_batches": args.limit_train_batches,
            "limit_val_batches": args.limit_val_batches,
            "limit_test_batches": args.limit_test_batches,
            "num_workers": args.num_workers,
        },
    )
    return resolved


def _build_data(dataset: str, config: ReSRConfig, num_workers: int = 0):
    data_dir = BASELINES_ROOT.joinpath("data")
    if dataset == "hatexplain":
        loader = build_loader_for_dataset(dataset=dataset, data_dir=data_dir)
        collator = GloVeEmbedderCollator(
            name=config.embedding_name,
            dim=config.embedding_dim,
            use_pretrained_only=config.use_pretrained_only,
            cache_dir=BASELINES_ROOT.joinpath("embeddings"),
        )
        extractor = HighlightExtractor()
    elif dataset == "toy":
        loader = build_loader_for_dataset(dataset=dataset, data_dir=data_dir)
        collator = OneHotEmbedderCollator()
        extractor = HighlightExtractor(
            text_splitter=lambda text: list(text),
            text_merger=lambda tokens: "".join(tokens),
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    train_df, val_df, test_df = loader.get_splits()
    collator.fit(df=train_df)
    train_data = _make_loader(train_df, config, collator, shuffle=True, num_workers=num_workers)
    val_data = _make_loader(val_df, config, collator, shuffle=False, num_workers=num_workers)
    test_data = _make_loader(test_df, config, collator, shuffle=False, num_workers=num_workers)
    return train_data, val_data, test_data, collator, extractor


def _make_loader(df, config: ReSRConfig, collator, *, shuffle: bool, num_workers: int = 0):
    dataset = HighlightDataset(
        texts=df.text.values,
        masks=df["mask"].values,
        labels=df.label.values,
        highlights=df.highlight.values,
        sample_ids=df.sample_id.values,
    )
    return DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=config.batch_size,
        collate_fn=collator,
        num_workers=num_workers,
    )


def _build_vrsr_hparams(config: ReSRConfig, resolved: dict) -> VRSRHyperparams:
    return VRSRHyperparams(
        scheduler_policy=resolved["scheduler_policy"],
        r_star=resolved["r_star"],
        epsilon=config.epsilon,
        alpha=config.alpha,
        beta=config.beta,
        gamma=config.gamma,
        delta=config.delta,
        eta=config.eta,
        mu=config.mu,
        nu=config.nu,
        q=config.q,
        warmup_epochs=config.warmup_epochs,
        inherit_gap_threshold=config.inherit_gap_threshold,
        inherit_patience=config.inherit_patience,
        inherit_slices=config.inherit_slices,
        probe_gap_threshold=resolved["probe_gap_threshold"],
        predictor_gap_threshold=resolved["predictor_gap_threshold"],
        gap_ema_decay=resolved["gap_ema_decay"],
        collapse_min_rate=resolved["collapse_min_rate"],
        collapse_max_rate=resolved["collapse_max_rate"],
        recovery_slices=config.recovery_slices,
        use_generator_scheduler_guidance=config.use_generator_scheduler_guidance,
        use_predictor_scheduler_guidance=config.use_predictor_scheduler_guidance,
    )


def _model_kwargs(*, config, collator, vrsr_hparams, resolved):
    return dict(
        model=_build_base_model(config, collator),
        num_classes=config.num_classes,
        optimizer_class=config.optimizer_class,
        optimizer_kwargs=config.optimizer_kwargs,
        vrsr_hparams=vrsr_hparams,
        probe_train_batches=resolved["probe_train_batches"],
        probe_train_epochs=resolved["probe_train_epochs"],
        probe_lr=resolved["probe_lr"],
        probe_init_from_online=resolved["probe_init_from_online"],
        probe_calibration_coefficient=resolved["probe_calibration_coefficient"],
        probe_calibration_temperature=resolved["probe_calibration_temperature"],
        probe_calibration_warmup_epochs=resolved["probe_calibration_warmup_epochs"],
        probe_calibration_ramp_epochs=resolved["probe_calibration_ramp_epochs"],
        probe_calibration_min_gain=resolved["probe_calibration_min_gain"],
        training_semantics=resolved["training_semantics"],
        manual_gradient_clip_val=resolved["manual_gradient_clip_val"],
        deterministic_budget_constraints=resolved["deterministic_budget_constraints"],
        per_example_budget=resolved["per_example_budget"],
        warmup_full_input=resolved["warmup_full_input"],
        mask_padding_constraints=resolved["mask_padding_constraints"],
        isolate_diagnostic_rng=resolved["isolate_diagnostic_rng"],
        collapse_recovery_budget=resolved["collapse_recovery_budget"],
        isolate_inactive_dropout=resolved["isolate_inactive_dropout"],
        diagnostic_seed=resolved["diagnostic_seed"],
        model_selection_rate_weight=resolved["model_selection_rate_weight"],
        model_selection_frag_weight=resolved["model_selection_frag_weight"],
        model_selection_min_f1=resolved["model_selection_min_f1"],
        model_selection_min_rate=resolved["model_selection_min_rate"],
        model_selection_clf_penalty=resolved["model_selection_clf_penalty"],
        teacher_alignment_coefficient=resolved["teacher_alignment_coefficient"],
        teacher_alignment_source=resolved["teacher_alignment_source"],
        teacher_alignment_mode=resolved["teacher_alignment_mode"],
        teacher_alignment_schedule=resolved["teacher_alignment_schedule"],
        teacher_alignment_warmup_epochs=resolved["teacher_alignment_warmup_epochs"],
        teacher_alignment_confidence_threshold=resolved["teacher_alignment_confidence_threshold"],
        teacher_alignment_require_correct=resolved["teacher_alignment_require_correct"],
        **_metrics(config.num_classes, include_val_highlights=config.has_validation_highlights),
    )


def _build_resr_model(
    *,
    config: ReSRConfig,
    collator,
    vrsr_hparams: VRSRHyperparams,
    train_data,
    val_data,
    resolved: dict,
):
    return ReSRHlModel(
        train_dataloader=train_data,
        val_dataloader=val_data,
        ref_train_batches=config.ref_train_batches,
        ref_train_epochs=config.ref_train_epochs,
        ref_lr=config.ref_lr,
        add_sparsity_loss=config.add_sparsity_loss,
        sparsity_level=config.sparsity_level,
        sparsity_coefficient=config.sparsity_coefficient,
        add_continuity_loss=config.add_continuity_loss,
        continuity_coefficient=config.continuity_coefficient,
        drift_coefficient=config.drift_coefficient,
        classification_coefficient=config.classification_coefficient,
        **_model_kwargs(
            config=config,
            collator=collator,
            vrsr_hparams=vrsr_hparams,
            resolved=resolved,
        ),
    )


def _load_resr_model(
    *,
    ckpt_filepath: Path,
    config: ReSRConfig,
    collator,
    vrsr_hparams: VRSRHyperparams,
    resolved: dict,
):
    return ReSRHlModel.load_from_checkpoint(
        checkpoint_path=ckpt_filepath,
        strict=False,
        **_model_kwargs(
            config=config,
            collator=collator,
            vrsr_hparams=vrsr_hparams,
            resolved=resolved,
        ),
    )


def _build_base_model(config: ReSRConfig, collator) -> MCD:
    return MCD(
        vocab_size=collator.vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_size=config.hidden_size,
        classification_head=config.classification_head,
        selection_head=config.selection_head,
        dropout_rate=config.dropout_rate,
        embedding_matrix=collator.embedding_matrix,
        temperature=config.temperature,
        freeze_embeddings=config.freeze_embeddings,
    )


def _metrics(num_classes: int, include_val_highlights: bool = True) -> dict:
    return {
        "val_classification_metrics": MetricCollection(
            {"f1": F1Score(task="multiclass", average="macro", num_classes=num_classes)}
        ),
        "test_classification_metrics": MetricCollection(
            {"f1": F1Score(task="multiclass", average="macro", num_classes=num_classes)}
        ),
        "val_highlight_metrics": MetricCollection(
            {
                "hl_pos_f1": BinaryHighlightF1Score(pos_label=1),
                "hl_rate": SelectionRate(),
                "hl_size": SelectionSize(),
            }
        )
        if include_val_highlights
        else None,
        "test_highlight_metrics": MetricCollection(
            {
                "hl_pos_f1": BinaryHighlightF1Score(pos_label=1),
                "hl_rate": SelectionRate(),
                "hl_size": SelectionSize(),
            }
        ),
    }


def _log_run_config(resolved: dict, config: ReSRConfig) -> None:
    logging.info("Scheduler policy: %s", resolved["scheduler_policy"])
    logging.info(
        "Rate control: target=%.4f, collapse=[%.4f, %.4f], structural checkpoint=[%.4f, %.4f]",
        resolved["r_star"],
        resolved["collapse_min_rate"],
        resolved["collapse_max_rate"],
        resolved["structural_min_rate"],
        resolved["structural_max_rate"],
    )
    logging.info("Training semantics: %s", resolved["training_semantics"])
    if resolved["teacher_alignment_coefficient"] > 0.0:
        logging.info(
            "Token-level teacher alignment enabled with coefficient=%.4f (source=%s, mode=%s)",
            resolved["teacher_alignment_coefficient"],
            resolved["teacher_alignment_source"],
            resolved["teacher_alignment_mode"],
        )
        if resolved["teacher_alignment_schedule"] != "constant":
            logging.info(
                "Teacher alignment schedule=%s with warmup_epochs=%d",
                resolved["teacher_alignment_schedule"],
                resolved["teacher_alignment_warmup_epochs"],
            )
        if (
            resolved["teacher_alignment_confidence_threshold"] > 0.0
            or resolved["teacher_alignment_require_correct"]
        ):
            logging.info(
                "Teacher alignment gate: confidence_threshold=%.4f, require_correct=%s",
                resolved["teacher_alignment_confidence_threshold"],
                resolved["teacher_alignment_require_correct"],
            )
    logging.info(
        "Probe fit config: batches=%d, epochs=%d, lr=%.4g",
        resolved["probe_train_batches"],
        resolved["probe_train_epochs"],
        resolved["probe_lr"],
    )
    logging.info(
        "Reference fit config: batches=%d (0 means full pass), epochs=%d, lr=%.4g",
        int(getattr(config, "ref_train_batches", 0)),
        int(getattr(config, "ref_train_epochs", 1)),
        float(getattr(config, "ref_lr", 1e-3)),
    )
    logging.info(
        "Selection config: rate_weight=%.4f, frag_weight=%.4f, min_f1=%.4f, min_rate=%.4f, clf_penalty=%.4f",
        resolved["model_selection_rate_weight"],
        resolved["model_selection_frag_weight"],
        resolved["model_selection_min_f1"],
        resolved["model_selection_min_rate"],
        resolved["model_selection_clf_penalty"],
    )
    logging.info(
        "Training semantics details: deterministic_budget=%s, per_example_budget=%s, "
        "mask_padding=%s, full_input_warmup=%s, isolate_diagnostic_rng=%s, "
        "collapse_recovery_budget=%s, isolate_inactive_dropout=%s, "
        "gradient_clip=%.2f, diagnostic_seed=%d",
        resolved["deterministic_budget_constraints"],
        resolved["per_example_budget"],
        resolved["mask_padding_constraints"],
        resolved["warmup_full_input"],
        resolved["isolate_diagnostic_rng"],
        resolved["collapse_recovery_budget"],
        resolved["isolate_inactive_dropout"],
        resolved["manual_gradient_clip_val"],
        resolved["diagnostic_seed"],
    )
    logging.info(
        "Checkpoint selection: monitor=%s, mode=%s",
        resolved["checkpoint_monitor"],
        resolved["checkpoint_mode"],
    )
    logging.info(
        "Scheduler guidance: generator=%s, predictor=%s",
        resolved["use_generator_scheduler_guidance"],
        resolved["use_predictor_scheduler_guidance"],
    )
    logging.info(
        "Scheduler gap filter: probe_threshold=%.4f, predictor_threshold=%.4f, ema_decay=%.4f",
        resolved["probe_gap_threshold"],
        resolved["predictor_gap_threshold"],
        resolved["gap_ema_decay"],
    )
    logging.info(
        "Probe calibration: coefficient=%.4f, temperature=%.4f, warmup=%d, ramp=%d, min_gain=%.4f, init_from_online=%s",
        resolved["probe_calibration_coefficient"],
        resolved["probe_calibration_temperature"],
        resolved["probe_calibration_warmup_epochs"],
        resolved["probe_calibration_ramp_epochs"],
        resolved["probe_calibration_min_gain"],
        resolved["probe_init_from_online"],
    )
    logging.info("Early stopping patience=%d", resolved["early_stopping_patience"])


def _summarize_and_save_metrics(
    metrics: dict, selected_seeds, seed_records, save_path: Path
) -> None:
    summary = {"selected_seeds": [int(seed) for seed in selected_seeds]}
    for split_name in ["validation", "test"]:
        summary[split_name] = {}
        metric_names = list(metrics[split_name].keys())
        for metric_name in metric_names:
            metric_values = np.array(metrics[split_name][metric_name]).reshape(
                len(selected_seeds), -1
            )
            per_seed_avg = metric_values.mean(axis=-1)
            per_seed_std = metric_values.std(axis=-1)
            avg = per_seed_avg.mean(axis=-1)
            std = per_seed_avg.std(axis=-1)
            metrics[split_name][f"per_seed_avg_{metric_name}"] = (
                per_seed_avg,
                per_seed_std,
            )
            metrics[split_name][f"avg_{metric_name}"] = (avg, std)
            summary[split_name][metric_name] = {
                "mean": float(avg),
                "std": float(std),
                "per_seed": {
                    str(seed): float(value) for seed, value in zip(selected_seeds, per_seed_avg)
                },
            }
    metrics["selected_seeds"] = np.asarray(selected_seeds, dtype=np.int64)
    metrics["per_seed_records"] = seed_records
    logging.info(metrics)
    np.save(save_path.joinpath("metrics.npy").as_posix(), metrics)
    _write_json_atomic(save_path.joinpath("summary_metrics.json"), summary)


def _metric_to_float(value) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _make_seed_record(
    *,
    seed: int,
    checkpoint: Path,
    checkpoint_monitor: str,
    validation_metrics: dict,
    test_metrics: dict,
) -> dict:
    record = {
        "seed": int(seed),
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_monitor": checkpoint_monitor,
    }
    for split_name, split_metrics in [
        ("validation", validation_metrics),
        ("test", test_metrics),
    ]:
        for metric_name, metric_value in sorted(split_metrics.items()):
            record[f"{split_name}.{metric_name}"] = _metric_to_float(metric_value)
    return record


def _write_seed_records(seed_records: list[dict], save_path: Path) -> None:
    _write_json_atomic(save_path.joinpath("per_seed_metrics.json"), seed_records)

    csv_path = save_path.joinpath("per_seed_metrics.csv")
    fieldnames = ["seed", "checkpoint", "checkpoint_monitor"]
    metric_fields = sorted(
        {key for record in seed_records for key in record if key not in fieldnames}
    )
    temporary_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames + metric_fields)
        writer.writeheader()
        writer.writerows(seed_records)
    temporary_path.replace(csv_path)


def _write_json_atomic(path: Path, payload) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_sample_ids(loader: DataLoader) -> list[int]:
    dataset = loader.dataset
    return [int(value) for value in np.asarray(dataset.sample_ids).reshape(-1)]


def _dataset_fingerprint(loader: DataLoader) -> dict:
    sample_ids = _dataset_sample_ids(loader)
    encoded = json.dumps(sample_ids, separators=(",", ":")).encode("utf-8")
    return {
        "examples": len(sample_ids),
        "sample_ids_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _write_reproducibility_manifest(
    *,
    save_path: Path,
    dataset: str,
    selected_seeds,
    loaders: dict[str, DataLoader],
    resolved: dict,
) -> None:
    source_paths = [
        Path(__file__).resolve(),
        RESR_ROOT.joinpath("config.py"),
        RESR_ROOT.joinpath("components", "resr.py"),
        RESR_ROOT.joinpath("modeling", "vrsr.py"),
        RESR_ROOT.joinpath("modeling", "stability.py"),
        BASELINES_ROOT.joinpath("components", "data_loader.py"),
        BASELINES_ROOT.joinpath("modeling", "losses.py"),
        BASELINES_ROOT.joinpath("modeling", "spp.py"),
    ]
    gpu_name = None
    if th.cuda.is_available():
        gpu_name = th.cuda.get_device_name(th.cuda.current_device())
    payload = {
        "dataset": dataset,
        "runtime": resolved["runtime"],
        "selected_seeds": [int(seed) for seed in selected_seeds],
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "pytorch_lightning": getattr(L, "__version__", "unknown"),
            "numpy": np.__version__,
            "torch": th.__version__,
        },
        "cuda": {
            "available": bool(th.cuda.is_available()),
            "version": th.version.cuda,
            "device": gpu_name,
        },
        "splits": {
            split_name: _dataset_fingerprint(loader) for split_name, loader in loaders.items()
        },
        "source_sha256": {
            path.relative_to(PROJECT_ROOT).as_posix(): _sha256(path) for path in source_paths
        },
        "resolved_config": {
            key: value
            for key, value in resolved.items()
            if isinstance(value, (str, int, float, bool, type(None)))
        },
    }
    _write_json_atomic(save_path.joinpath("reproducibility.json"), payload)
    _write_json_atomic(
        save_path.joinpath("split_sample_ids.json"),
        {split_name: _dataset_sample_ids(loader) for split_name, loader in loaders.items()},
    )


def _list_checkpoints(seed_ckpt_path: Path):
    return sorted(
        seed_ckpt_path.glob("*.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _clear_checkpoints(seed_ckpt_path: Path) -> None:
    for checkpoint in seed_ckpt_path.glob("*.ckpt"):
        checkpoint.unlink()


def _clear_stale_run_outputs(save_path: Path, selected_seeds) -> None:
    for filename in [
        "metrics.npy",
        "summary_metrics.json",
        "per_seed_metrics.json",
        "per_seed_metrics.csv",
        "reproducibility.json",
        "split_sample_ids.json",
    ]:
        path = save_path.joinpath(filename)
        if path.exists():
            path.unlink()
    for seed in selected_seeds:
        prediction_path = save_path.joinpath(f"predictions_seed={int(seed)}.csv")
        if prediction_path.exists():
            prediction_path.unlink()


def _lightning_limit_value(value: float):
    if value >= 1 and float(value).is_integer():
        return int(value)
    return value
