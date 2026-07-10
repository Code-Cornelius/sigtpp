# Paper Experiments

This directory contains the main experiment runner, config loader, dataset wiring, and output logic used for the
paper-style training runs.

## Config Resolution

`load_experiment_config(...)` in `training_helpers.py` resolves configs as follows:

1. `"<dataset>/<model>.yaml"` first tries the shared root-level file `configs/<model>.yaml`.
2. If the basename does not exist at the root, it falls back to the exact dataset-local path under
   `configs/<dataset>/...`.
3. If `configs/<dataset>/experiment.yaml` exists, it is merged in automatically before returning the final config.

This means most runs use the shared root-level model YAMLs:

- `ddpm.yaml`
- `wgan.yaml`
- `sigtpp.yaml`
- `vae.yaml`
- `deter.yaml`
- `gamma.yaml`

## Available Experiments

The 9 datasets used in the paper:

| Paper | Experiment name           |
|-------|---------------------------|
| PS    | `poisson_three_marks`     |
| IP    | `inh_poisson_three_marks` |
| H1    | `hawkes`                  |
| H3    | `hawkes_3x3`              |
| EQ    | `earthquake`              |
| SO    | `stackoverflow`           |
| TB    | `taobao`                  |
| TX    | `taxi`                    |
| YLP   | `yelp_mississauga`        |


Experiment registration is driven by `test/paper_experiments/settings/*.py` plus the side-effect imports at the bottom
of [trainingmanager.py](trainingmanager.py).

## Troubleshooting

- Import errors usually mean the script was not launched from the repo root with `src/` added to `sys.path`.
- `Config Not Found` means the path passed to `load_experiment_config(...)` does not match the shared-root or
  dataset-local rules above.
- `Checkpoint Not Found` usually means no validation step finished, so no `model-*.ckpt` was written.
- For new datasets, the most common missing step is forgetting the side-effect import in `trainingmanager.py`.
