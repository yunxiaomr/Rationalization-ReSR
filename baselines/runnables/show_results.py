import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MAIN_SEEDS = [2023, 1337, 12345, 31415, 42]
BASELINE_MODELS = {"fr", "mgr", "mcd", "grat"}
DATASET_ORDER = {
    "toy": 0,
    "hatexplain": 1,
}
MODEL_ORDER = {
    "fr": 0,
    "mgr": 1,
    "mcd": 2,
    "grat": 3,
    "genspp": 4,
    "resr": 5,
}
GENSPP_RUN_PREFIX = {
    "toy": ["genspp_toy_final"],
    "hatexplain": ["genspp_hatexplain_final"],
}
GENSPP_MAX_LEN = {
    "toy": 20,
    "hatexplain": 30,
}


METRIC_NAMES = [
    "f1",
    "hl_pos_f1",
    "hl_rate",
    "hl_size",
    "SP",
    "CT",
]
METRICS_INFO = {
    "f1": {"is_percentage": True},
    "hl_pos_f1": {"is_percentage": True},
    "hl_size": {"is_percentage": False},
    "hl_rate": {"is_percentage": True},
    "SP": {"is_percentage": False},
    "CT": {"is_percentage": False},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize baseline results, including GenSPP from baselines/genspp."
    )
    parser.add_argument(
        "--all-methods",
        action="store_true",
        help="Include the main ReSR run and compute paper BRS for complete six-method comparisons.",
    )
    parser.add_argument(
        "--no-genspp",
        action="store_true",
        help="Do not append GenSPP results from baselines/genspp/results/outputs.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional path to save the summary table as CSV.",
    )
    return parser.parse_args()


def parse_metric(metric: str, metric_name: str):
    if metric == "-":
        return metric
    if metric_name == "BRS":
        return f"${float(metric):.2f}$"

    is_percentage = METRICS_INFO[metric_name]["is_percentage"]
    mean, std = metric.split(" +/- ")
    mean = float(mean)
    std = float(std)
    parsed_metric = (
        f"${mean * 100 if is_percentage else mean:.2f}"
        + "_{\\pm "
        + f"{std * 100 if is_percentage else std:.2f}"
        + "}$"
    )
    return parsed_metric


def format_results(df):
    formatted = [
        " & ".join(
            [
                parse_metric(metric=value, metric_name=key)
                for key, value in row.to_dict().items()
                if key not in ["dataset", "model", "seeds"]
            ]
        )
        for _, row in df.iterrows()
    ]
    df["Formatted"] = formatted
    return df


def format_mean_std(mean, std):
    return f"{float(mean):.4f} +/- {float(std):.4f}"


def append_metric_row(results, dataset_name, model_name, metric_values):
    results.setdefault("dataset", []).append(dataset_name)
    results.setdefault("model", []).append(model_name)
    results.setdefault("seeds", []).append(
        ",".join(str(int(seed)) for seed in metric_values.get("seeds", []))
    )
    for metric_name in METRIC_NAMES:
        results.setdefault(metric_name, []).append(metric_values.get(metric_name, "-"))


def collect_gradient_baselines(results_path: Path, models=None):
    models = BASELINE_MODELS if models is None else models
    rows = []
    for results_file in results_path.rglob("metrics.npy"):
        dataset_name = results_file.parent.parent.name
        model_name = results_file.parent.name

        if model_name.split("__", 1)[0] not in models:
            continue

        metrics = np.load(results_file.as_posix(), allow_pickle=True).item()
        metric_values = {}
        for metric_name in METRIC_NAMES:
            avg_key = f"avg_test_{metric_name}"
            if avg_key in metrics.get("test", {}):
                mean, std = metrics["test"][avg_key]
                metric_values[metric_name] = format_mean_std(mean, std)
            else:
                metric_values[metric_name] = "-"
        metric_values["seeds"] = list(metrics.get("selected_seeds", []))
        label = "resr" if model_name == "resr_main" else model_name
        rows.append((dataset_name, label, metric_values))
    return rows


def read_genspp_metric_file(path: Path):
    metrics = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    return metrics


def read_genspp_mask_stats(mask_path: Path):
    sizes = []
    rates = []
    original_len = None
    for line in mask_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("Original:"):
            original = line.split("Original:", 1)[1].strip()
            original_len = len(original.split())
        elif line.startswith("Masked:"):
            selected = line.split("Masked:", 1)[1].strip()
            selected_len = 0 if not selected else len(selected.split())
            sizes.append(selected_len)
            if original_len is not None and original_len > 0:
                rates.append(selected_len / original_len)
            original_len = None
    if not sizes:
        return None, None
    return (
        float(np.mean(rates)) if rates else None,
        float(np.mean(sizes)),
    )


def collect_genspp(genspp_root: Path):
    outputs_root = genspp_root.joinpath("results", "outputs")
    if not outputs_root.exists():
        return []

    rows = []
    for dataset_name, prefixes in GENSPP_RUN_PREFIX.items():
        metric_files = []
        mask_files = []
        summary_records = []
        for prefix in prefixes:
            summary_path = outputs_root.joinpath(f"metrics_summary_{prefix}.json")
            if summary_path.exists():
                payload = json.loads(summary_path.read_text(encoding="utf-8-sig"))
                summary_records.extend(payload.get("records", []))
            else:
                metric_files.extend(outputs_root.glob(f"metrics_{prefix}_seed=*.txt"))
                mask_files.extend(outputs_root.glob(f"masks_{prefix}_seed=*.txt"))
        metric_files = sorted(metric_files)
        mask_files = sorted(mask_files)
        if not metric_files and not summary_records:
            continue

        clf_f1 = []
        hl_f1 = []
        rates = []
        sizes = []
        sp = []
        ct = []
        metric_dicts = [record.get("metrics", {}) for record in summary_records]
        metric_dicts.extend(read_genspp_metric_file(metric_file) for metric_file in metric_files)
        for metrics in metric_dicts:
            if "Macro F1 Classification" in metrics:
                clf_f1.append(metrics["Macro F1 Classification"])
            elif "Binary Micro F1 Classification" in metrics:
                clf_f1.append(metrics["Binary Micro F1 Classification"])
            if "Binary Micro F1 Mask" in metrics:
                hl_f1.append(metrics["Binary Micro F1 Mask"])
            if "Sparsity" in metrics:
                sp.append(metrics["Sparsity"])
            if "Contiguity" in metrics:
                ct.append(metrics["Contiguity"])
            if "Selection Rate" in metrics:
                rates.append(metrics["Selection Rate"])
            if "Selection Size" in metrics:
                sizes.append(metrics["Selection Size"])

        if not rates or not sizes:
            parsed_rates = []
            parsed_sizes = []
            for mask_file in mask_files:
                rate, size = read_genspp_mask_stats(mask_file)
                if size is not None:
                    parsed_sizes.append(size)
                if rate is not None:
                    parsed_rates.append(rate)
            if not rates:
                rates = parsed_rates
            if not sizes:
                sizes = parsed_sizes

        if not rates and sp:
            rates = list(sp)
        if not sizes and rates:
            max_len = GENSPP_MAX_LEN[dataset_name]
            sizes = [rate * max_len for rate in rates]

        metric_values = {
            "seeds": [record["seed"] for record in summary_records],
            "f1": mean_std_or_dash(clf_f1),
            "hl_pos_f1": mean_std_or_dash(hl_f1),
            "hl_rate": mean_std_or_dash(rates),
            "hl_size": mean_std_or_dash(sizes),
            "SP": mean_std_or_dash(sp),
            "CT": mean_std_or_dash(ct),
        }
        rows.append((dataset_name, "genspp", metric_values))

    return rows


def mean_std_or_dash(values):
    if not values:
        return "-"
    return format_mean_std(np.mean(values), np.std(values))


def add_brs(frame):
    frame = frame.copy()
    frame["BRS"] = "-"
    metrics = ("f1", "hl_pos_f1", "hl_rate", "hl_size")
    for dataset, group in frame.groupby("dataset"):
        if len(group) != 6 or set(group["model"]) != set(MODEL_ORDER):
            print(f"BRS omitted for {dataset}: one result for each of the six methods is required.")
            continue
        if "seeds" not in group or any(
            sorted(str(seeds).split(",")) != sorted(str(seed) for seed in MAIN_SEEDS)
            for seeds in group["seeds"]
        ):
            print(
                f"BRS omitted for {dataset}: all methods must use the five main-experiment seeds."
            )
            continue
        if any(value == "-" for name in metrics for value in group[name]):
            print(f"BRS omitted for {dataset}: one or more metrics are missing.")
            continue
        values = np.array(
            [
                [float(row[name].split(" +/- ")[0]) for name in metrics]
                for _, row in group.iterrows()
            ]
        )
        if not np.isfinite(values).all():
            print(f"BRS omitted for {dataset}: one or more metrics are non-finite.")
            continue
        values[:, :3] *= 100.0
        values = values.round(2)
        low, high = values.min(axis=0), values.max(axis=0)
        span = np.maximum(high - low, 1e-12)
        normalized = (values - low) / span
        normalized[:, 2:] = (high[2:] - values[:, 2:]) / span[2:]
        scores = 100 * normalized @ np.array([0.30, 0.50, 0.10, 0.10])
        frame.loc[group.index, "BRS"] = [f"{value:.2f}" for value in scores]
    return frame


if __name__ == "__main__":
    args = parse_args()
    baselines_root = Path(__file__).parent.parent.resolve()
    results_path = baselines_root.joinpath("results")
    genspp_root = baselines_root.joinpath("genspp")

    results = {}
    rows = collect_gradient_baselines(results_path)
    if not args.no_genspp:
        rows.extend(collect_genspp(genspp_root))
    if args.all_methods:
        rows = [row for row in rows if row[1] in MODEL_ORDER]
        rows.extend(
            row
            for row in collect_gradient_baselines(
                baselines_root.parent / "resr" / "results", models={"resr_main"}
            )
            if row[1] == "resr"
        )

    rows = sorted(
        rows,
        key=lambda item: (
            DATASET_ORDER.get(item[0], 99),
            MODEL_ORDER.get(item[1], 99),
            item[1],
        ),
    )
    for dataset_name, model_name, metric_values in rows:
        append_metric_row(results, dataset_name, model_name, metric_values)

    df = pd.DataFrame.from_dict(results)
    if df.empty:
        print("No matching results found.")
        raise SystemExit(0)

    if args.all_methods:
        df = add_brs(df)
    df = format_results(df)
    if args.output_csv is not None:
        output_path = Path(args.output_csv).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved summary to {output_path}")

    print(df.to_string(index=False))
