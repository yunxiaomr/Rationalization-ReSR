# GenSPP

Adapted from [nlp-unibo/gen-spp](https://github.com/nlp-unibo/gen-spp).
Install the dependencies and prepare the external files using the
[root README](../../README.md). This implementation shares
`baselines/data/` with the other methods.

Run from this directory:

```bash
python -m pip install -r ../requirements.txt

python train.py toy genspp_toy_final
python train.py hatexplain genspp_hatexplain_final
```

Default seeds are `2023 1337 12345 31415 42`; override them with `--seeds`.
`--w` sets CPU workers and `--cpu` forces CPU execution. The default search uses
100 generations and a population of 50. Settings are in `src/utils/config.py`.

For a short execution check:

```bash
python train.py toy smoke --seeds 2023 --generations 1 --population-size 2 --cpu
```

Non-default search budgets add a suffix such as `__generations1_population2`
to the run name. For HateXplain, the reader merges training and validation
inputs before sampling GenSPP's internal validation subset.

Weights are saved to `results/weights/`; masks, metrics, and seed-keyed JSON
summaries are saved to `results/outputs/`.

To fine-tune a saved predictor and evaluate it:

```bash
python evaluate.py toy genspp_toy_final_seed=2023 --e 5 --seed 2023
```

`--e 0` evaluates saved weights without fine-tuning; the default is five epochs.
`--seed` controls validation sampling and fine-tuning. Evaluation overwrites the
model's text metrics and masks; use these outputs separately from the training
JSON summary.
