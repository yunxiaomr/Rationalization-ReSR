import logging
from pathlib import Path

import pytorch_lightning as L
import numpy as np
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification.f_beta import F1Score

from components.callbacks import PycharmProgressBar
from components.data_loader import ToyLoader
from components.metrics import BinaryHighlightF1Score, SelectionRate, SelectionSize
from components.model import MCDHlModel
from components.processing import (
    HighlightDataset,
    OneHotEmbedderCollator,
    HighlightExtractor,
)
from configurations.base import ConfigKey
from configurations.model import MCDConfig
from modeling.spp import MCD
from runnables.benchmark_cli import (
    run_path,
    list_checkpoints,
    parse_runner_args,
    resolve_selected_seeds,
)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_runner_args("Train or evaluate MCD on the Toy dataset.")

    save_path = Path(__file__).parent.parent.resolve().joinpath("results", "toy", "mcd")
    save_path = run_path(save_path, args)
    if not save_path.exists():
        save_path.mkdir(parents=True)

    ckpt_path = save_path.joinpath("checkpoints")
    if not ckpt_path.exists():
        ckpt_path.mkdir(parents=True)

    data_dir = Path(__file__).parent.parent.resolve().joinpath("data")

    config = MCDConfig.from_config(key=ConfigKey(dataset="toy", tags={"spp", "mcd"}))
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    selected_seeds = resolve_selected_seeds(config.seeds, args.seeds)

    trainer_args = {
        "accelerator": "auto",
        "devices": 1,
        "accumulate_grad_batches": 1,
        "max_epochs": args.max_epochs if args.max_epochs is not None else config.max_epochs,
    }
    for key in ("limit_train_batches", "limit_val_batches", "limit_test_batches"):
        value = getattr(args, key)
        if value is not None:
            trainer_args[key] = int(value) if value >= 1 and value.is_integer() else value

    seed_everything(seed=15000)

    loader = ToyLoader(data_dir=data_dir)
    train_df, val_df, test_df = loader.get_splits()

    collator = OneHotEmbedderCollator()
    collator.fit(df=train_df)

    train_data = HighlightDataset(
        texts=train_df.text.values,
        masks=train_df["mask"].values,
        labels=train_df.label.values,
        highlights=train_df.highlight.values,
        sample_ids=train_df.sample_id.values,
    )
    train_data = DataLoader(
        train_data,
        shuffle=True,
        batch_size=config.batch_size,
        collate_fn=collator,
        num_workers=args.num_workers,
    )

    val_data = HighlightDataset(
        texts=val_df.text.values,
        masks=val_df["mask"].values,
        labels=val_df.label.values,
        highlights=val_df.highlight.values,
        sample_ids=val_df.sample_id.values,
    )
    val_data = DataLoader(
        val_data,
        batch_size=config.batch_size,
        collate_fn=collator,
        num_workers=args.num_workers,
    )

    test_data = HighlightDataset(
        texts=test_df.text.values,
        masks=test_df["mask"].values,
        labels=test_df.label.values,
        highlights=test_df.highlight.values,
        sample_ids=test_df.sample_id.values,
    )
    test_data = DataLoader(
        test_data,
        batch_size=config.batch_size,
        collate_fn=collator,
        num_workers=args.num_workers,
    )

    extractor = HighlightExtractor(
        text_splitter=lambda t: list(t), text_merger=lambda t: "".join(t)
    )

    metrics = {}
    for seed in selected_seeds:
        seed_everything(seed=seed)

        seed_ckpt_path = ckpt_path.joinpath(f"seed={seed}")
        seed_ckpt_path.mkdir(parents=True, exist_ok=True)

        model = MCD(
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

        model = MCDHlModel(
            model=model,
            num_classes=config.num_classes,
            optimizer_class=config.optimizer_class,
            optimizer_kwargs=config.optimizer_kwargs,
            val_classification_metrics=MetricCollection(
                {"f1": F1Score(task="multiclass", average="macro", num_classes=3)}
            ),
            test_classification_metrics=MetricCollection(
                {"f1": F1Score(task="multiclass", average="macro", num_classes=3)}
            ),
            val_highlight_metrics=MetricCollection(
                {
                    "hl_pos_f1": BinaryHighlightF1Score(pos_label=1),
                    "hl_rate": SelectionRate(),
                    "hl_size": SelectionSize(),
                }
            ),
            test_highlight_metrics=MetricCollection(
                {
                    "hl_pos_f1": BinaryHighlightF1Score(pos_label=1),
                    "hl_rate": SelectionRate(),
                    "hl_size": SelectionSize(),
                }
            ),
            add_sparsity_loss=config.add_sparsity_loss,
            sparsity_level=config.sparsity_level,
            sparsity_coefficient=config.sparsity_coefficient,
            add_continuity_loss=config.add_continuity_loss,
            continuity_coefficient=config.continuity_coefficient,
            classification_coefficient=config.classification_coefficient,
        )

        existing_ckpts = list_checkpoints(seed_ckpt_path)
        should_train = args.force_retrain or not existing_ckpts

        if should_train:
            csv_logger = CSVLogger(save_dir=str(save_path / "training_logs"), name=f"seed={seed}")
        else:
            csv_logger = None

        trainer = L.Trainer(
            **trainer_args,
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", patience=30),
                ModelCheckpoint(monitor="val_loss", mode="min", dirpath=seed_ckpt_path),
                PycharmProgressBar(),
            ],
            deterministic=True,
            logger=csv_logger,
        )

        if should_train:
            if args.force_retrain and existing_ckpts:
                logging.info(
                    f"Force retrain enabled for seed {seed}; existing checkpoints will be ignored."
                )
            trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)
            if trainer.checkpoint_callback.best_model_path:
                ckpt_filepath = Path(trainer.checkpoint_callback.best_model_path)
            else:
                latest_ckpts = list_checkpoints(seed_ckpt_path)
                if not latest_ckpts:
                    raise FileNotFoundError(
                        f"No checkpoint was produced for seed {seed} in {seed_ckpt_path}."
                    )
                ckpt_filepath = latest_ckpts[0]
        else:
            logging.info("Found existing checkpoint! Skipping model training...")
            ckpt_filepath = existing_ckpts[0]

        model = MCDHlModel.load_from_checkpoint(
            checkpoint_path=ckpt_filepath,
            model=model.model,
            val_classification_metrics=MetricCollection(
                {"f1": F1Score(task="multiclass", average="macro", num_classes=3)}
            ),
            test_classification_metrics=MetricCollection(
                {"f1": F1Score(task="multiclass", average="macro", num_classes=3)}
            ),
            val_highlight_metrics=MetricCollection(
                {
                    "hl_pos_f1": BinaryHighlightF1Score(pos_label=1),
                    "hl_rate": SelectionRate(),
                    "hl_size": SelectionSize(),
                }
            ),
            test_highlight_metrics=MetricCollection(
                {
                    "hl_pos_f1": BinaryHighlightF1Score(pos_label=1),
                    "hl_rate": SelectionRate(),
                    "hl_size": SelectionSize(),
                }
            ),
        )
        seed_everything(seed=seed)

        val_metrics = trainer.validate(model=model, dataloaders=val_data)[0]
        logging.info(f"Validation metrics: {val_metrics}")

        model.enable_storing_predictions()

        test_metrics = trainer.test(model=model, dataloaders=test_data)[0]
        test_predictions = model.test_predictions
        logging.info(f"Test metrics: {test_metrics}")

        for metric_name, metric_value in val_metrics.items():
            metrics.setdefault("validation", {}).setdefault(metric_name, []).append(metric_value)
        for metric_name, metric_value in test_metrics.items():
            metrics.setdefault("test", {}).setdefault(metric_name, []).append(metric_value)

        extractor(
            predictions=test_predictions,
            vocabulary=collator.vocabulary,
            save_path=save_path,
            seed=seed,
        )

        model.flush_predictions()
        model.disable_storing_predictions()

    for split_name in ["validation", "test"]:
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

    metrics["selected_seeds"] = np.asarray(selected_seeds, dtype=np.int64)
    logging.info(metrics)
    np.save(save_path.joinpath("metrics.npy").as_posix(), metrics)
