"""Print the contents of a per-replicate bootstrap .npz file.

Usage (editor run button):  set _PATH below and run directly.
Usage (CLI):
    python -c "import sys; sys.path.insert(0, 'src'); exec(open('test/paper_experiments/print_npz.py').read())" \
        -- path/to/file.npz
"""

import glob
import os
import sys
from typing import List, Tuple

import numpy as np

_NPZ = "out/taxi/results_on_test_npz/deter_final_test_2026-06-27_16-00-03.npz"
_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "test/paper_experiments"
_PATH = os.path.join(_HERE, _NPZ)
_DEFAULT_PATTERNS = [
    os.path.join(_HERE, "out", "*", "results_on_test_npz", "*.npz"),
    os.path.join(_HERE, "out", "*", "results_on_ablation", "*.npz"),
    os.path.join(_HERE, "out", "*", "results_on_multiseed", "*.npz"),
]


def _default_paths() -> List[str]:
    if os.path.exists(_PATH):
        return [_PATH]

    candidates = [p for pattern in _DEFAULT_PATTERNS for p in glob.glob(pattern)]
    if candidates:
        return [max(candidates, key=os.path.getmtime)]

    raise FileNotFoundError(
        f"No default .npz found. Set _NPZ to an existing file or pass one on the CLI. Patterns: {_DEFAULT_PATTERNS}"
    )


def _metric_std(valid: np.ndarray) -> float:
    if len(valid) <= 1:
        return 0.0
    return float(valid.std(ddof=1))


def _load_npz(path: str) -> Tuple[int, int, List[str], List[str], np.ndarray]:
    with np.load(path, allow_pickle=True) as npz:
        schema = int(npz["schema_version"])
        if schema != 1:
            raise ValueError(f"{path}: unsupported schema_version={schema}, expected 1")

        B = int(npz["B"])
        model_names = [str(x) for x in npz["model_names"].tolist()]
        metric_names = [str(x) for x in npz["metric_names"].tolist()]
        data = np.asarray(npz["data"], dtype=float)

    expected_shape = (len(model_names), len(metric_names), B)
    if data.shape != expected_shape:
        raise ValueError(f"{path}: data shape {data.shape}, expected {expected_shape}")

    return schema, B, model_names, metric_names, data


_argv = [a for a in sys.argv[1:] if a != "--"]
paths = _argv if _argv else _default_paths()

for path in paths:
    schema, B, model_names, metric_names, data = _load_npz(path)

    print(f"File   : {path}")
    print(f"Schema : v{schema}")
    print(f"B      : {B}")
    print(f"Models : {len(model_names)}")
    print(f"Metrics: {len(metric_names)}")
    print()

    col_w = max(len(m) for m in metric_names + ["metric"]) + 2
    header = f"{'metric':<{col_w}}  {'mean':>12}  {'std':>12}  {'min':>12}  {'max':>12}"
    for mi, model in enumerate(model_names):
        print(f"=== {model} ===")
        print(header)
        print("-" * len(header))
        for ki, metric in enumerate(metric_names):
            vec = data[mi, ki, :]
            valid = vec[~np.isnan(vec)]
            if len(valid) == 0:
                print(f"{metric:<{col_w}}  {'NaN':>12}  {'NaN':>12}  {'NaN':>12}  {'NaN':>12}")
            else:
                print(
                    f"{metric:<{col_w}}  {valid.mean():>12.6g}  {_metric_std(valid):>12.6g}"
                    f"  {valid.min():>12.6g}  {valid.max():>12.6g}"
                )
            print(f"{'':<{col_w}}  values: {np.array2string(vec, separator=', ')}")
        print()
