# ReSR: Responsibility-Aware Scheduling for Selective Rationalization

ReSR uses a virtual-runtime scheduler (VRSR) to allocate updates between the
rationale generator and predictor, guided by training-only probe and reference
diagnostics.

Main experiments on **Toy** and **HateXplain**, with ReSR and the FR, MGR, MCD,
G-RAT, and GenSPP baselines.

## Structure

```text
resr/          ReSR, VRSR, and main-experiment configurations
baselines/     FR, MGR, MCD, G-RAT, and GenSPP
```

## Installation

Use **Python 3.10**. Tested on Linux with PyTorch `1.13.1` and TorchText `0.14.1`.
Both subdirectories use the same pinned dependencies.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r resr/requirements.txt
```

Training selects an available accelerator automatically and saves logs locally.

For CPU-only Linux x86-64, install the matching PyTorch and TorchText CPU
wheels first, then install the remaining dependencies:

```bash
python -m pip install --no-deps --index-url https://download.pytorch.org/whl/cpu \
  torch==1.13.1+cpu torchtext==0.14.1
python -m pip install -r resr/requirements.txt
```

## Data

Datasets, embeddings, checkpoints, and results are excluded from this repository.

We use the processed datasets distributed with
[GenSPP](https://github.com/nlp-unibo/gen-spp/tree/main/genetic/data):

| Resource | Download | Local path |
| --- | --- | --- |
| Toy | [toy_dataset.pkl](https://raw.githubusercontent.com/nlp-unibo/gen-spp/main/genetic/data/toy_dataset.pkl) | `baselines/data/toy/toy_sequential_dataset_easy.pkl` |
| HateXplain training split | [train.pkl](https://raw.githubusercontent.com/nlp-unibo/gen-spp/main/genetic/data/hatexplain/train.pkl) | `baselines/data/hatexplain/train.pkl` |
| HateXplain validation split | [val.pkl](https://raw.githubusercontent.com/nlp-unibo/gen-spp/main/genetic/data/hatexplain/val.pkl) | `baselines/data/hatexplain/val.pkl` |
| HateXplain test split | [test.pkl](https://raw.githubusercontent.com/nlp-unibo/gen-spp/main/genetic/data/hatexplain/test.pkl) | `baselines/data/hatexplain/test.pkl` |
| Twitter GloVe | [glove.twitter.27B.zip](https://nlp.stanford.edu/data/glove.twitter.27B.zip) | `baselines/data/glove/twitter/glove.twitter.27B.25d.txt` |

Rename the Toy file as shown above. HateXplain uses GenSPP's processed splits of
the [official dataset](https://github.com/hate-alert/HateXplain); extract the
**25-dimensional Twitter GloVe** file to its listed path. Toy uses one-hot
features and needs no GloVe.
See the [data protocol](baselines/README.md#data-protocol) for preprocessing details.

## ReSR

Run commands from the repository root:

```bash
python -m resr.runnables.toy_resr_train
python -m resr.runnables.hatexplain_resr_train
```

Both use the paper configuration and five seeds:
`2023 1337 12345 31415 42`. Add `--seeds 2023` for a single run or
`--force-retrain` to retrain. See [resr/README.md](resr/README.md) for training settings,
checkpoint selection, and short execution checks.

## Baselines

From the repository root, run FR, MGR, MCD, or G-RAT:

```bash
cd baselines
python -m runnables.toy_fr_train
python -m runnables.hatexplain_fr_train
cd ..
```

Replace `fr` with `mgr`, `mcd`, or `grat` for the corresponding method.
See [baselines/README.md](baselines/README.md) for training options and data details.

Run GenSPP from the repository root:

```bash
cd baselines/genspp
python train.py toy genspp_toy_final
python train.py hatexplain genspp_hatexplain_final
cd ../..
```

GenSPP uses the same five seeds by default. See its
[README](baselines/genspp/README.md) for search settings and evaluation.

## Results

Outputs are saved to `resr/results/`, `baselines/results/`, and
`baselines/genspp/results/` for the respective methods.

From the repository root:

```bash
python -m resr.runnables.show_results --dataset all --split test
python -m baselines.runnables.show_results --all-methods
```

## Acknowledgments and license

The baseline implementations and shared preprocessing are adapted from
[GenSPP](https://github.com/nlp-unibo/gen-spp/tree/2b447a2f341c51714d16a85431cc1998159bf8f2)
([Ruggeri and Signorelli, ACL 2025](https://aclanthology.org/2025.acl-long.59/)).
`baselines/genspp/` follows its `genetic/` implementation. The original copyright
notice and MIT license are preserved in [LICENSE](LICENSE).
