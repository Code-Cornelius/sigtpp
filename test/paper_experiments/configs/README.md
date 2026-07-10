# Configuration Files

This directory contains the YAML configs used by `test/paper_experiments/train_model.py`.

## Layout

```text
configs/
|- ddpm.yaml
|- wgan.yaml
|- sigtpp.yaml
|- vae.yaml
|- deter.yaml
|- gamma.yaml
|- *_test.yaml
`- <dataset>/
   `- experiment.yaml   # optional dataset-specific values
```

The shared root-level YAMLs define model families. Optional `configs/<dataset>/experiment.yaml` files provide
dataset-specific values such as `time_max`, `adjacency`, or `mark_probs`.

## How Loading Works

`load_experiment_config(...)` supports two practical forms:

1. `"<dataset>/<model>.yaml"`
   Loads the shared root-level file `configs/<model>.yaml`.
2. `"<dataset>/<file>.yaml"`
   If `configs/<file>.yaml` does not exist, falls back to the exact dataset-local file under `configs/<dataset>/...`.

In both cases:

- `experiment_type` is injected from the leading path component
- if `configs/<dataset>/experiment.yaml` exists, it is merged first
- model YAML values win on key conflicts

## Common Top-Level Keys

- `version`: one of `ddpm`, `wgan`, `sigtpp`, `vae`, `deter`, `gamma`
- `seeds`: list of random seeds; use one value for a normal single-seed run or multiple values for multi-seed training
- `gpu_id`: list of GPU ids for PyTorch Lightning
- `epochs`: max epochs
- `patience`: early-stopping patience
- `period_log`: validation frequency
- `period_plotting_in_logs`: how often the custom logger updates plotted metrics
- `verbose`: console/log verbosity
- `server_training`: selects the sparse internal validation-plot schedule used by `TrainingManager`
- `diagnostic_only`: skip training and only evaluate an existing checkpoint
- `skip_diagnostics`: train but skip the final test/diagnostic phase
- `output_dir`: base directory; the code appends `out/<dataset>/...`
- `parameter_sets`: hyperparameter grid
- `n_bootstraps`: number of bootstrap replicates used when computing test metrics; **required** — the loader
  asserts its presence and will raise an error if it is absent. Production configs use `100`; smoke-test
  configs use `5`. Set to `1` to disable bootstrapping.
- `refine_best_n_bootstraps`: optional override applied only to the winning hyperparameter config during the
  final evaluation pass. Used by `sigtpp` configs to run more replicates on the selected model without
  inflating cost during the grid search. Omit or set to `null` to skip.

## Important Flag Semantics

### `seeds`

Use `seeds` for both single-seed and multi-seed runs:

```yaml
seeds: [ 0, 1, 2, 3, 4 ]
```

For a normal single-seed run, use `seeds: [ 42 ]`. When multiple seeds are provided, each seed is run with
`bootstrap_replicates: 1` and writes to an isolated `seed_<k>` output root. The final multi-seed summary uses explicit
across-seed columns such as `hist_it_seed_mean`, `hist_it_seed_std`, and `hist_it_seed_n_valid`; these `*_seed_std`
columns are standard deviations across trained seeds, not bootstrap standard deviations.

### `verbose`

Controls progress bars and console output. It is an explicit config value; it is not automatically derived from
`server_training`.

### `server_training`

This does not change the trainer itself. It mainly changes the internal `period_plot_val` passed into model factories:

- local mode:
    - `deter`: 1
    - `gamma`: 100
  - `ddpm`, `sigtpp`, `wgan`, `vae`: 500
- server mode:
    - all current architectures: 100000

### `diagnostic_only`

Skips training and attempts to load the best existing checkpoint for the run directory.

### `skip_diagnostics`

Used mostly by smoke tests. Training runs, but the final evaluation/test phase and its plotting are skipped.

## `output_dir`

The default configs use:

```yaml
output_dir: "test/paper_experiments"
```

The code then writes under:

```text
test/paper_experiments/out/<dataset>/
|- results_on_val_txt/
|- results_on_test_txt/
|- results_on_test_npz/
|- results_on_ablation/        # sig-degree ablation report + per-replicate vectors
|- results_on_multiseed/       # all per-seed + aggregate output (when `seeds` has >1 entry)
`- models/
```

See the Outputs section in the top-level `README.md` for the per-file layout.

## `parameter_sets`

`parameter_sets` defines a grid search. Each Cartesian-product combination becomes one run.

Example:

```yaml
parameter_sets:
  lr: [ 1.0e-4, 1.0e-5 ]
  hid_size_rnn: [ 16, 32 ]
  concentration_factor: [ 1.0 ]
```

For `sigtpp`, specify exactly one of:

- `sig_degree`: an absolute signature degree.
- `relative_sig_degree`: an offset from the degree detector's largest usable degree. The detector considers degrees up
  to 10 by default, so if it finds 8, offsets `[ -2, -1, 0, 1 ]` resolve to training degrees `[ 6, 7, 8, 9 ]`.
  Relative configs resolving below 3 or above 10 are skipped.
