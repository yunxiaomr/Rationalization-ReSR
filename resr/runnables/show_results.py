import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


RESR_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RESR_ROOT.parent
BASELINES_ROOT = PROJECT_ROOT.joinpath("baselines")
if str(BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINES_ROOT))


DEFAULT_METRICS = {
    "validation": ["f1", "hl_pos_f1", "hl_rate", "hl_size", "loss", "select_score"],
    "test": ["f1", "hl_pos_f1", "hl_rate", "hl_size", "loss"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize ReSR main-experiment results from saved metrics.npy files."
    )
    parser.add_argument(
        "--dataset",
        choices=["toy", "hatexplain", "all"],
        default="all",
        help="Filter by dataset. Defaults to all ReSR result directories.",
    )
    parser.add_argument(
        "--split",
        choices=["validation", "test", "both"],
        default="test",
        help="Which split to display. Defaults to test.",
    )
    parser.add_argument(
        "--include-short-runs",
        action="store_true",
        help="Include runs made with an explicit training-budget override.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional path to save the summary table as CSV.",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default=None,
        help="Optional result root. Defaults to resr/results.",
    )
    parser.add_argument(
        "--per-seed",
        action="store_true",
        help="Display seed-keyed metrics instead of only aggregate mean/std.",
    )
    return parser.parse_args()


def load_run_config(result_dir: Path):
    config_path = result_dir.joinpath("run_config.json")
    if not config_path.exists():
        return None
    return json.loads(config_path.read_text(encoding="utf-8-sig"))


def format_metric(metrics, split_name: str, metric_name: str):
    split_metrics = metrics.get(split_name, {})
    prefix = "val" if split_name == "validation" else "test"
    avg_key = f"avg_{prefix}_{metric_name}"
    if avg_key in split_metrics:
        mean, std = split_metrics[avg_key]
        return f"{float(mean):.4f} +/- {float(std):.4f}", float(mean)
    return "-", float("-inf")


def _load_seed_records(metrics: dict, result_dir: Path, run_config) -> list[dict]:
    records_path = result_dir.joinpath("per_seed_metrics.json")
    if records_path.exists():
        return json.loads(records_path.read_text(encoding="utf-8-sig"))
    if metrics.get("per_seed_records"):
        return list(metrics["per_seed_records"])

    seeds = metrics.get("selected_seeds")
    if seeds is None and run_config is not None:
        seeds = run_config.get("selected_seeds")
    if seeds is None:
        return []
    seeds = [int(seed) for seed in np.asarray(seeds).reshape(-1)]

    records = [{"seed": seed} for seed in seeds]
    for split_name, split_metrics in [
        ("validation", metrics.get("validation", {})),
        ("test", metrics.get("test", {})),
    ]:
        for metric_name, values in split_metrics.items():
            if not metric_name.startswith("per_seed_avg_"):
                continue
            per_seed_values = values[0] if isinstance(values, tuple) else values
            per_seed_values = np.asarray(per_seed_values).reshape(-1)
            if len(per_seed_values) != len(seeds):
                continue
            raw_metric_name = metric_name.removeprefix("per_seed_avg_")
            for record, value in zip(records, per_seed_values):
                record[f"{split_name}.{raw_metric_name}"] = float(value)
    return records


def collect_rows(
    results_root: Path,
    dataset_filter: str,
    split_filter: str,
    include_short_runs: bool = False,
    per_seed: bool = False,
):
    rows = []
    split_names = ["validation", "test"] if split_filter == "both" else [split_filter]

    for metrics_path in results_root.rglob("metrics.npy"):
        relative = metrics_path.relative_to(results_root)
        dataset_name = relative.parts[0]
        if dataset_filter != "all" and dataset_name != dataset_filter:
            continue

        relative_run = "/".join(relative.parts[1:-1])
        if relative_run != "resr_main" and not (
            include_short_runs and relative_run.startswith("resr_main__")
        ):
            continue

        metrics = np.load(metrics_path.as_posix(), allow_pickle=True).item()
        run_config = load_run_config(metrics_path.parent)
        run_label = Path(relative_run).name

        if per_seed:
            seed_records = _load_seed_records(metrics, metrics_path.parent, run_config)
            for record in seed_records:
                for split_name in split_names:
                    prefix = "val" if split_name == "validation" else "test"
                    row = {
                        "dataset": dataset_name,
                        "run": run_label,
                        "folder": relative_run,
                        "split": split_name,
                        "seed": int(record["seed"]),
                        "path": metrics_path.parent.as_posix(),
                    }
                    for metric_name in DEFAULT_METRICS[split_name]:
                        value = record.get(f"{split_name}.{prefix}_{metric_name}")
                        row[metric_name] = "-" if value is None else f"{float(value):.4f}"
                    row["_sort_hl_pos_f1"] = float(
                        record.get(f"{split_name}.{prefix}_hl_pos_f1", float("-inf"))
                    )
                    rows.append(row)
            continue

        for split_name in split_names:
            metric_names = DEFAULT_METRICS[split_name]
            row = {
                "dataset": dataset_name,
                "run": run_label,
                "folder": relative_run,
                "split": split_name,
                "path": metrics_path.parent.as_posix(),
            }
            sort_value = float("-inf")
            for metric_name in metric_names:
                metric_value, raw_value = format_metric(metrics, split_name, metric_name)
                row[metric_name] = metric_value
                if metric_name == "hl_pos_f1":
                    sort_value = raw_value
            row["_sort_hl_pos_f1"] = sort_value
            rows.append(row)

    return rows


if __name__ == "__main__":
    args = parse_args()
    results_root = (
        Path(args.results_root).expanduser().resolve()
        if args.results_root is not None
        else RESR_ROOT.joinpath("results")
    )

    rows = collect_rows(
        results_root=results_root,
        dataset_filter=args.dataset,
        split_filter=args.split,
        include_short_runs=args.include_short_runs,
        per_seed=args.per_seed,
    )

    if not rows:
        print("No matching ReSR metrics were found.")
        raise SystemExit(0)

    df = pd.DataFrame(rows)
    sort_columns = ["dataset", "split", "_sort_hl_pos_f1"]
    df = df.sort_values(by=sort_columns, ascending=[True, True, False]).drop(
        columns=["_sort_hl_pos_f1"]
    )

    if args.output_csv is not None:
        output_path = Path(args.output_csv).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved summary to {output_path}")

    print(df.to_string(index=False))
