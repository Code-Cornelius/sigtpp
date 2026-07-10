# Datasets

All dataset modules live under `test/paper_experiments/data/`. Each is a
`pytorch_lightning.LightningDataModule` subclass; the training pipeline constructs
them via the `data_factory` registered in `settings/<name>.py`.

This public release ships the nine datasets used in the paper: four synthetic
(`poisson_three_marks`, `inh_poisson_three_marks`, `hawkes`, `hawkes_3x3`) and five
real-world (`taxi`, `stackoverflow`, `taobao`, `earthquake`, `yelp_mississauga`).

## Marks support

All data modules expose the canonical mark tensor contract. Some datasets carry
real categorical marks; unmarked or single-mark datasets use the trivial
all-zero mark tensors.

- Marks are stored in `self.train_marks`, `self.val_marks`, `self.test_marks`.
- Shape: `(N, L+1)`, dtype `long`, 0-indexed categories.
- Column 0 is the anchor position (not a prediction target). For **synthetic datasets** (Poisson,
  Hawkes, Hawkes3x3) a zero mark is explicitly prepended, so column 0 is always `0`. For
  **EasyTPP datasets** (taxi, stackoverflow, taobao, earthquake) sequences are normalised
  so the first real event is at t=0; no synthetic anchor is prepended, so column 0 holds the mark
  of the first real event (any category 0…K-1). The shape and `inputs_len` convention are the same
  in both cases; only the semantic of position 0 differs.
- Dataloaders return the canonical 3-tuple `(inputs, inputs_len, marks)` for every dataset.
- Unmarked or single-mark datasets still provide `marks`, using the trivial all-zero tensor contract.
- Whether marks are actually used by a training run depends on the experiment setting wiring, not just the data module.

## Synthetic datasets

### `PoissonDataModule`

**Path:** `synthetic/poisson/poisson_dataset.py`
**Cache:** `data/synthetic/<filename>.pt` (auto-generated, gitignored)

| Experiment                | `use_IHP_or_HP` | `base_intensity` | T    | data_size | num_marks | mark_probs                             |
|---------------------------|-----------------|------------------|------|-----------|-----------|----------------------------------------|
| `poisson_three_marks`     | False           | 1.0              | 12.0 | 2000      | 3         | YAML-set (default `[0.7, 0.05, 0.25]`) |
| `inh_poisson_three_marks` | True            | 1.0              | 10.0 | 5000      | 3         | YAML-set (default `[0.7, 0.05, 0.25]`) |

**IHP reproducibility caveat:** `ihp.gen` uses `np.random.rand` (global legacy RNG) for
inter-arrival times and a seeded `np.random.Generator` for marks. Full reproducibility of
IHP datasets requires controlling the global numpy seed before construction.

### `HawkesDataModule`

**Path:** `synthetic/hawkes/hawkes_dataset.py`
Univariate Hawkes process (H1), run as a single-class (unmarked) process:
`num_marks = 1`, all-zero mark tensors, and the dataloader emits the canonical 3-tuple.

### `Hawkes3x3DataModule`

**Path:** `synthetic/hawkes/hawkes_3x3_dataset.py`
**Cache:** `data/synthetic/<filename>.pt` (auto-generated, gitignored)

Tick-backed 3-node multivariate Hawkes process. K=3 node streams are merged
into one nondecreasing marked stream; the originating node id (0, 1, or 2) is
the categorical mark. A time anchor at 0.0 and anchor mark 0 are prepended.

| Experiment   | Nodes | `baseline`      | `time_max` | `data_size` | num_marks |
|--------------|-------|-----------------|------------|-------------|-----------|
| `hawkes_3x3` | 3     | [0.5, 0.5, 0.5] | 15.0       | 2000        | 3         |

Adjacency and decay matrices are specified in `configs/hawkes_3x3/experiment.yaml` and merged
automatically by `load_experiment_config(...)`.

## Real datasets

| Experiment         | Module              | Path            | Marks                            |
|--------------------|---------------------|-----------------|----------------------------------|
| `taxi`             | `EasyTPPDataModule` | `real/easytpp/` | Yes (categorical)                |
| `stackoverflow`    | `EasyTPPDataModule` | `real/easytpp/` | Yes                              |
| `taobao`           | `EasyTPPDataModule` | `real/easytpp/` | Yes                              |
| `earthquake`       | `EasyTPPDataModule` | `real/easytpp/` | Yes                              |
| `yelp_mississauga` | `EditTPPDataModule` | `real/editpp/`  | Trivial single-class (all zeros) |

### EasyTPP datasets

**Path:** `real/easytpp/`
**Source:** Hugging Face datasets loaded as `load_dataset("easytpp/<name>")`.
**Cache:** `data/easytpp/<dataset>/`

The active loader is `EasyTPPDataModule`. It stores cumulative-time tensors with shape
`(N, L+1, 1)` and mark tensors with shape `(N, L+1)`, using the Hugging Face split names
`train`, `validation`, and `test`.

### EDITPP datasets

EDITPP datasets are stored as `.pkl` files under `data/editpp/` and loaded via `torch.load`.
Each file is a dict with `sequences` (list of per-sequence dicts), `t_max`, and `mean_number_items`.
Each sequence dict has an `arrival_times` key holding cumulative event times (numpy array, no anchor).

The `EditTPPDataModule` base class:

- Prepends a 0.0 anchor to each sequence, producing the canonical `(N, L+1, 1)` shape.
- Drops anchor-only sequences, then shuffles the flat `sequences` list with a seed before splitting.
- Splits 60/20/20 (train/val/test).
- Jitters any zero inter-arrivals per split for numerical safety.
- Provides trivial all-zero mark tensors (`num_marks=1`).

Wired EDITPP dataset (canonical Add-Thin name; Lüdke et al., NeurIPS 2023):

| Experiment         | Add-Thin name / paper id | `.pkl` file            | Notes    |
|--------------------|--------------------------|------------------------|----------|
| `yelp_mississauga` | Yelp2 (Mississauga)      | `yelp_mississauga.pkl` | uncapped |
