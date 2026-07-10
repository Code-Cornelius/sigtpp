# Tests

## Structure

```text
tests_src/
|- test_data/         # Dataset loaders and data-contract tests
|- test_generators/   # Synthetic generator tests
|- test_metrics/      # Metric correctness tests
|- test_nn/           # Architecture, training, and registry tests
|- test_utils/        # Utility and preprocessing tests
|- conftest.py
`- test_*.py          # Standalone top-level tests
```

## CI

GitHub Actions runs:

```bash
python -m pytest tests_src/ -v --tb=long
```

with:

- Python 3.8
- CPU-only Torch 1.12.1
- `PYTHONPATH=src:.`

Tests guarded by `pytest.importorskip("signatory")` skip automatically in CI, since `signatory` is not installed there.

### Not run in CI

| Path                                                  | Reason                                                                          |
|-------------------------------------------------------|---------------------------------------------------------------------------------|
| `test_nn/test_experiment_registry.py`                 | imports `trainingmanager` → `TPPArchitecture` → `tppmetrics` → `signatory`      |
| `test_nn/test_forward_include_first_it.py`            | lazy import of `TPPArchitecture` inside test function                           |
| `test_nn/test_mark_sampling_contract.py`              | imports `TPPArchitecture`                                                       |
| `test_nn/test_metrics_cum_comparable_across_modes.py` | imports `tppmetrics` directly                                                   |
| `test_nn/test_preprocess_dataset_for_metrics.py`      | imports `TPPArchitecture`                                                       |
| `test_nn/test_sample_for_fixed_batch.py`              | imports `TPPArchitecture`                                                       |
| `test_nn/test_scale_paths_consistency.py`             | imports `TPPArchitecture`                                                       |
| `test_nn/test_smoke_dataset_model_matrix.py`          | imports `trainingmanager` → `TPPArchitecture`                                   |
| `test_nn/test_training_manager_integration.py`        | imports `trainingmanager`                                                       |
| `test_nn/test_model_pruning.py`                       | imports `trainingmanager` → `tppmetrics` → `sigw1metric_exp` → `signatory`      |
| `test_nn/test_training_manager_unit.py`               | imports `trainingmanager`                                                       |
| `test_nn/test_training_runner.py`                     | imports `training_runner` → `trainingmanager` → `TPPArchitecture` → `signatory` |
| `test_nn/test_train_model_sigtpp.py`                  | imports `trainingmanager`                                                       |
| `test_nn/test_train_seq_cap.py`                       | imports `ArchitectureOneToOne` → `tppmetrics` → `signatory`                     |
| `test_utils/test_tpp_utils.py`                        | imports `TPPArchitecture`                                                       |
| `test_metrics/test_sigw1metric.py`                    | directly imports `signatory`                                                    |
| `test_nn/test_architecture_vae.py`                    | imports `Architecture_VAE` → `TPPArchitecture` → `tppmetrics` → `signatory`     |
| `test_nn/test_marks_wiring_regressions.py`            | imports `Architecture_DDPM` → `TPPArchitecture` → `tppmetrics` → `signatory`    |

## Running Locally

```bash
# Linux/macOS
PYTHONPATH=src:. python -m pytest tests_src/ -v

# Windows PowerShell
$env:PYTHONPATH="src;."
python -m pytest tests_src/ -v
```

If you have a matching `signatory` build available, the signatory-dependent `test_nn/` and
`test_metrics/test_sigw1metric.py` coverage will run as well.

## Notes

- `test_nn/test_smoke_dataset_model_matrix.py` is the smoke matrix for dataset/model combinations.
- Real-world data tests live in `test_data/`, including the recent EDITPP dataset-contract coverage.
