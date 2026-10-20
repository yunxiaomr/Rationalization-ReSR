# ReSR

Run the main experiments from the repository root:

```bash
python -m resr.runnables.toy_resr_train
python -m resr.runnables.hatexplain_resr_train
```

Both use the five seeds `2023 1337 12345 31415 42`. Add `--seeds 2023` for a
single run or `--force-retrain` to replace the selected seeds' checkpoints.

## Training and selection

The configurations in `config.py` specify the main experiments. Training uses
stochastic Gumbel masks, a global batch budget, the original padding treatment,
and rationale-only predictor warmup.

Checkpoints maximize `val_select_score`, which combines validation rationale
quality, rationale rate, fragmentation, and a classification penalty. This
selection rule uses validation rationale annotations. Test annotations are
used only for evaluation.

Implementation: `modeling/vrsr.py` (scheduler), `components/resr.py` (training),
and `config.py` (dataset settings).

## Short execution check

```bash
python -m resr.runnables.toy_resr_train --seeds 2023 --max-epochs 2 \
  --limit-train-batches 1 --limit-val-batches 1 --limit-test-batches 1
```

Budget overrides use separate result directories, for example
`resr_main__epochs2_train1p0_val1p0_test1p0`. Settings are recorded in
`run_config.json`. Use `--help` for runtime options.

## Reference results

Paper results, reported as mean ± standard deviation across five seeds:

| Dataset | Clf-F1 (%) | Hl-F1 (%) | Rate (%) | Size |
| --- | --- | --- | --- | --- |
| Toy | 99.95 ± 0.08 | 85.47 ± 9.04 | 14.77 ± 0.38 | 2.95 ± 0.08 |
| HateXplain | 72.92 ± 1.11 | 42.30 ± 2.71 | 16.99 ± 2.38 | 2.28 ± 0.34 |

## Results

Outputs are saved under `resr/results/<dataset>/resr_main/`, including
checkpoints, aggregate and per-seed metrics, predictions, configurations,
reproducibility records, and CSV logs.

Summaries cover the requested seeds. Rerun the default command to aggregate all
five seeds; existing checkpoints are reused.

```bash
python -m resr.runnables.show_results --dataset all --split test
python -m resr.runnables.show_results --dataset toy --split test --per-seed
```

Add `--include-short-runs` to display results from reduced training budgets.
