# Baselines

FR, MGR, MCD, and G-RAT share the loaders, metrics, and configurations in this
directory. GenSPP is in [`genspp/`](genspp/README.md).

Prepare the datasets and embeddings using the [root README](../README.md).
Run prior methods from this directory:

```bash
python -m pip install -r requirements.txt

python -m runnables.toy_fr_train
python -m runnables.toy_mgr_train
python -m runnables.toy_mcd_train
python -m runnables.toy_grat_train
```

Replace `toy` with `hatexplain` to run the corresponding HateXplain entrypoint.
The default seeds are `2023 1337 12345 31415 42`; `--seeds` accepts an explicit
list. Models use the published global-budget loss and padding treatment.
Checkpoints are selected by minimum validation loss.

Use `--force-retrain` to replace existing checkpoints. Run an entrypoint with
`--help` for epoch, batch-size, batch-limit, and worker options.

Training metrics and checkpoints are written to `results/<dataset>/<method>/`.
Logs are local CSV files under each method's `training_logs/` directory.
Explicit training-budget overrides add a directory suffix so short checks
cannot reuse or replace the default experiment's checkpoint.

```bash
python -m runnables.show_results
```

To compare all six formal methods, including ReSR, and compute the paper's
Balanced Rationalization Score:

```bash
python -m runnables.show_results --all-methods --output-csv paper_results.csv
```

BRS normalizes the six methods within each dataset and uses weights
`(0.30, 0.50, 0.10, 0.10)` for Clf-F1, Hl-F1, rate, and size. Rate and size
are reversed so lower is better. Means are rounded to the two decimal places
displayed in the paper before normalization. BRS requires all four metrics for
all six methods using the five main-experiment seeds. Custom-budget runs are
excluded.

## Data protocol

Download links and expected local paths are in the [root README](../README.md#data).
ReSR and the baselines use the same shared loaders.

**Toy.** The released file contains 10,000 sequences of 20 characters, three
classes, and character-level rationale annotations. The baseline methods
use 6,400 training, 1,600 validation, and 2,000 test examples, with fixed
validation-sampling seed `15000`. Rename the downloaded `toy_dataset.pkl` to
`toy_sequential_dataset_easy.pkl`. The file's SHA-256 is
`2ee223d8aecd6ee9a585aa695a6fd5fd6f4c10c39fe6046ffb833d7ab44fb064`.

**HateXplain.** The processed GenSPP splits contain 15,383 / 1,922 / 1,924 posts
before filtering. Loaders retain posts with at most 30 tokens. Annotator labels
`hatespeech=0` and `offensive=2` are merged into class `0` before majority voting;
`normal=1` remains class `1`. ReSR and the baselines combine rationale
masks by strict majority vote. GenSPP retains its original handling of multiple
annotator masks and internal validation sampling.

**Embeddings.** HateXplain uses the 25-dimensional Twitter GloVe vectors from
[Stanford NLP](https://nlp.stanford.edu/projects/glove/). Extract
`glove.twitter.27B.25d.txt` under `data/glove/twitter/`. The baseline
pipeline also builds a cache in `embeddings/`. Toy uses one-hot features.
