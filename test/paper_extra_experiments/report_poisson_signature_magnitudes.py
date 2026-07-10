r"""
Poisson signature magnitude diagnostic.

Question
--------
For a Poisson process N(t) with constant rate lambda = 1 on [0, T], how do
the magnitudes (mean, std) of E[Sig_k((t, N(t)))] evolve as k grows from 1
to MAX_DEGREE, and at what depth does the per-term std drop below the
relevant numerical noise floor?

Why
---
Two questions, both pragmatic:

(a) The signature-theory bound |S^(k)| <= L^k / k! is a *decay* bound only
    once k > L. For a raw Poisson path with T=12, lambda=1, L ~ 24, so the
    factorial only dominates beyond k ~ 24 -- much further than the max
    depth (10) the codebase ever uses. We print L^k/k! alongside the
    observed magnitudes so the regime is explicit.

(b) SigW1DegreeDetector operates on the *standardised* path (post
    StandardScaler + /total_vars), where L is O(1) per channel and the
    factorial decay does kick in within k=1..10. We mimic that pipeline
    on the Poisson realisations and report the same statistics there.

Also: detector threshold is 1e-8 absolute, while float32 eps ~= 1.2e-7.
The script prints both dtypes so any boundary fragility is visible.

Usage
-----
    python -c "import sys; sys.path.insert(0, 'src'); exec(open('test/paper_extra_experiments/report_poisson_signature_magnitudes.py').read())"

LaTeX explainer (NeurIPS style)
-------------------------------

\section{Signature values, batch std, and the unusability cliff}

\paragraph{Signature.} For a bounded-variation path $X:[0,T]\to\mathbb{R}^d$
(here $d=2$: time channel and count channel) the level-$k$ signature is the
iterated integral
\begin{equation}
    S^{(k)}(X)_{0,T}
    \;=\;
    \int_{0<t_1<\cdots<t_k<T}\!
    dX_{t_1}\otimes\cdots\otimes dX_{t_k}\;\in\;\mathbb{R}^{d^k}.
    \label{eq:sig}
\end{equation}
At depth $k$ there are $d^k$ scalar coordinates. \texttt{signatory.signature(x,
depth=K)} returns the concatenation $(S^{(1)},\dots,S^{(K)})$ of length
$\sum_{k=1}^{K} d^k$.

\paragraph{Factorial bound.} The standard rough-path estimate
\begin{equation}
    \bigl\|S^{(k)}(X)_{0,T}\bigr\|
    \;\le\;
    \frac{\|X\|^{k}_{1\text{-var};[0,T]}}{k!}
    \;=\;\frac{L^{k}}{k!}
    \label{eq:bound}
\end{equation}
is a \emph{decay} bound only once $k>L$: below that, $L/k>1$ and the
right-hand side is still growing. For the raw Poisson path $(t, N(t))$ on
$[0,12]$ with $\lambda=1$, $L\approx 24$, so depths $1{-}10$ live in the
growth regime of~\eqref{eq:bound}. Decay only dominates past $k\sim 24$.

\paragraph{Two distinct std's.} Given a batch of $N$ realisations, let
$\hat\sigma_j$ be the empirical standard deviation of signature coordinate
$j\in\{1,\dots,\sum_k d^k\}$ across the batch. The same quantity plays
\emph{two} roles in the pipeline:
\begin{enumerate}
    \item \textbf{Detector.} \texttt{SigW1DegreeDetector} flags depth $k$ as
    \emph{dead} when
    \(
        \hat\sigma_j < 10^{-8}\ \text{for every}\ j\ \text{at depth}\ k.
    \)
    A dead depth carries no distributional signal across the dataset.
    \item \textbf{Standardisation in the loss.} When
    \texttt{standardise=True}, \texttt{SigW1MetricExp} builds a per-coordinate
    rescaler $z_j\mapsto z_j/\max(\hat\sigma_j,10^{-8})$ before computing the
    squared-mean-difference. The clamp at $10^{-8}$ guards against division
    blow-up.
\end{enumerate}

\paragraph{The loss.} With $\bar S_{\text{real}}^{(j)}$ and
$\bar S_{\text{gen}}^{(j)}$ denoting batch means, the SigW1 loss is
\begin{equation}
    \mathcal{L}_{\mathrm{SigW1}}
    \;=\;
    \sum_{j}
    \frac{
        \bigl(\bar S_{\text{real}}^{(j)} - \bar S_{\text{gen}}^{(j)}\bigr)^{2}
    }{
        \max(\hat\sigma_j,\,10^{-8})^{2}
    }.
    \label{eq:loss}
\end{equation}

\paragraph{When does a coordinate become unusable?}
Three regimes, sorted by $\hat\sigma_j$:
\begin{itemize}
    \item \emph{Healthy} ($\hat\sigma_j$ moderate): coordinate contributes a
    meaningful gradient.
    \item \emph{Tiny $\hat\sigma_j<10^{-8}$ (dead):} the clamp activates;
    $(\bar S_{\text{real}}^{(j)} - \bar S_{\text{gen}}^{(j)})^{2}$ is just
    round-off, divided by $10^{-16}$, so the term emits amplified numerical
    noise into the gradient. \emph{This is what \texttt{effective\_sig\_degree}
    trims.}
    \item \emph{Large $\hat\sigma_j$:} on the raw Poisson path at depth $10$,
    coordinate magnitudes reach $\sim 10^{6}$. Standardisation divides them by
    a comparable $\hat\sigma_j$, so the contribution to~\eqref{eq:loss} ends
    up of order one. \emph{Large raw terms are not unusable -- they are
    downweighted.}
\end{itemize}

\paragraph{Why the detector works \emph{after} standardisation.}
\texttt{SigW1DegreeDetector} is fed the standardised path
$\widetilde X \,=\,\bigl(X-\mu\bigr)/(\hat\sigma \cdot \|\text{paths}\|_{TV})$
(\texttt{StandardScaler + /total\_vars}). On $\widetilde X$, the 1-variation
collapses to $L_{\widetilde X}\approx 1$, so
\(\,L_{\widetilde X}^{k}/k! \to 2.8\times 10^{-7}\) at $k=10$, and the
factorial bound is well inside detector territory. The script confirms this
empirically: every one of the $1024$ coordinates at depth $10$ has
$\hat\sigma_j < 10^{-8}$, and $155$ of $512$ at depth $9$ do too.

\paragraph{Trim rule.} With dead set $\mathcal{D}\subseteq\{1,\dots,K\}$,
let $m=\max\mathcal{D}$ and define the trailing dead block as the maximal
contiguous run $\{k,k+1,\dots,m\}\subseteq\mathcal{D}$. The detector trims
from the \emph{start} of that block:
\begin{equation}
    k^{\star}
    \;=\;
    \max\!\left(1,\;
        \min\bigl\{\,k\in\{1,\dots,m\}\,:\,\{k,k+1,\dots,m\}\subseteq\mathcal{D}\bigr\}\;-\;1
    \right).
\end{equation}
We do \emph{not} require dead degrees to form a contiguous tail of
$\{1,\dots,K\}$: alive degrees \emph{above} the trailing block are
treated as noise that escaped the threshold and dropped together with
the tail; alive degrees \emph{below} the trailing block are kept even
if some intermediate degree is dead. Examples ($K=10$):
$\mathcal{D}=\{8,9,10\}\Rightarrow k^{\star}=7$;
$\mathcal{D}=\{4,7\}\Rightarrow k^{\star}=6$ (trim from $7$, keep $1$--$6$
including the isolated dead $4$);
$\mathcal{D}=\{4,7,8,9,10\}\Rightarrow k^{\star}=6$ (same: trim from $7$).

\paragraph{Float32 caveat.} The detector threshold $10^{-8}$ sits below
$\varepsilon_{f32}\approx 1.19\times 10^{-7}$, so on the \emph{raw}
deterministic time-axis coordinate at depth $1$, float64 reports
$\hat\sigma=0$ (dead) while float32 reports $\hat\sigma\approx 5\times
10^{-8}$ (alive). The discrepancy is confined to the boundary of the raw
regime: in the standardised regime the two dtypes agree on the verdict.
"""

import logging
import math
import sys
from typing import Dict, List, Tuple

import torch

from src.logger.init_logger import set_config_logging

set_config_logging()
logger = logging.getLogger(__name__)

try:
    import signatory
except ImportError:
    print("signatory is not installed - cannot compute signatures.")
    sys.exit(1)


# ---- configuration ---------------------------------------------------------
LAMBDA = 1.0
T_MAX = 12.0
N_SEQS = 2000
MAX_DEGREE = 10
SEED = 42
FLOAT32_EPS = torch.finfo(torch.float32).eps  # ~1.19e-7
FLOAT64_EPS = torch.finfo(torch.float64).eps  # ~2.22e-16
DETECTOR_THR = 1e-8  # threshold used inside SigW1DegreeDetector

PATH_DIM = 2  # (t, N(t))


def simulate_poisson_paths(
    n_seqs: int, t_max: float, lam: float, dtype: torch.dtype, seed: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (paths, eff_lens). paths: (n_seqs, L, 2); eff_lens: (n_seqs,) long.

    Channel 0 = time, channel 1 = count. Sequences are padded with constant-end:
    after the last event both channels stay constant up to t_max. `eff_lens[i]`
    is the number of non-padded positions for path i (anchor + events + terminal),
    so callers can mask padding when computing statistics -- mirroring how the
    codebase uses `variable_len_standard_stats` over `effective_lens`.
    """
    gen = torch.Generator().manual_seed(seed)
    counts = torch.poisson(torch.full((n_seqs,), lam * t_max, dtype=torch.float64), generator=gen).long()
    max_events = int(counts.max().item())
    L = max_events + 2  # anchor + events + terminal at t_max

    paths = torch.zeros((n_seqs, L, 2), dtype=dtype)
    eff_lens = counts + 2  # anchor + k events + terminal anchor at t_max
    for i in range(n_seqs):
        k = int(counts[i].item())
        if k == 0:
            paths[i, 1:, 0] = t_max
            paths[i, 1:, 1] = 0.0
            continue
        u = torch.rand(k, generator=gen).to(dtype)
        event_times, _ = torch.sort(u * t_max)
        paths[i, 1 : 1 + k, 0] = event_times
        paths[i, 1 : 1 + k, 1] = torch.arange(1, k + 1, dtype=dtype)
        # terminal anchor + constant-end pad
        paths[i, 1 + k :, 0] = t_max
        paths[i, 1 + k :, 1] = float(k)
    return paths, eff_lens


def _valid_mask(paths: torch.Tensor, eff_lens: torch.Tensor) -> torch.Tensor:
    """Boolean (N, L) mask: True where position is a non-padded path value."""
    L = paths.shape[1]
    idx = torch.arange(L, device=paths.device).unsqueeze(0)
    return idx < eff_lens.unsqueeze(1)


def per_degree_slice(sig: torch.Tensor, deg: int, paths_dim: int) -> torch.Tensor:
    """Return the slice of the flat signature tensor corresponding to depth `deg`."""
    ptr = 0
    for d in range(1, deg):
        ptr += paths_dim**d
    n = paths_dim**deg
    return sig[..., ptr : ptr + n]


def summarise(paths: torch.Tensor, max_degree: int, eps_factor_threshold: float) -> Tuple[List[Dict], torch.Tensor]:
    """Compute per-degree summary stats. Returns (rows, full_sig).

    Asserts signatory preserves input dtype. Counts coordinates dead under both
    the hard 1e-8 threshold (what the codebase uses) and a dtype-aware
    `eps_factor_threshold` (e.g. 10 * dtype_eps).
    """
    sig = signatory.signature(paths, depth=max_degree)
    assert sig.dtype == paths.dtype, f"signatory downcast {paths.dtype} -> {sig.dtype}"
    rows: List[Dict] = []
    for d in range(1, max_degree + 1):
        block = per_degree_slice(sig, d, paths.shape[2])  # (N, paths_dim**d)
        mean_per_term = block.mean(0)
        std_per_term = block.std(0)
        rows.append(
            {
                'degree': d,
                'n_terms': block.shape[1],
                'abs_mean_max': mean_per_term.abs().max().item(),
                'abs_mean_median': mean_per_term.abs().median().item(),
                'std_max': std_per_term.max().item(),
                'std_median': std_per_term.median().item(),
                'std_min': std_per_term.min().item(),
                'n_dead_1e8': int((std_per_term < DETECTOR_THR).sum().item()),
                'n_dead_10eps': int((std_per_term < eps_factor_threshold).sum().item()),
            }
        )
    return rows, sig


def total_variation_per_path(paths: torch.Tensor) -> torch.Tensor:
    """Per-path sum-norm 1-variation: sum over channels of sum_t |dx_t|. Shape (N,)."""
    dx = paths[:, 1:] - paths[:, :-1]
    return dx.abs().sum(dim=(1, 2))


def standardise_in_dtype(
    paths64: torch.Tensor, eff_lens: torch.Tensor, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Faithful StandardScaler + /total_vars done end-to-end in the target dtype.

    Mean/std are computed over *non-padded* positions only, matching the codebase's
    `variable_len_standard_stats(seqs, effective_lens)`. The final divide-by-
    total-variation uses the per-path mean L computed in the same dtype.
    """
    p = paths64.to(dtype)
    mask = _valid_mask(p, eff_lens)  # (N, L)
    flat = p[mask]  # (sum(eff_lens), 2)
    mean = flat.mean(0)
    std = flat.std(0).clamp_min(torch.finfo(dtype).eps)
    p = (p - mean) / std
    L_per_path = total_variation_per_path(p)
    L_mean = L_per_path.mean().clamp_min(torch.finfo(dtype).eps)
    p = p / L_mean
    return p, total_variation_per_path(p)


def print_table(title: str, rows: List[Dict], eps: float, L_stats: Dict[str, float]) -> None:
    L_med = L_stats['median']
    print(
        f"\n=== {title}  (eps = {eps:.3e}, "
        f"10*eps = {10 * eps:.3e}, "
        f"L per-path median = {L_med:.2f}, min = {L_stats['min']:.2f}, max = {L_stats['max']:.2f}) ==="
    )
    header = (
        f"{'deg':>3} {'n_terms':>7} {'L_med^k/k!':>14} {'|mean|_max':>12} "
        f"{'std_max':>12} {'std_med':>12} {'std_min':>12} "
        f"{'n<1e-8':>8} {'n<10*eps':>10}"
    )
    print(header)
    print('-' * len(header))
    for r in rows:
        bound = L_med ** r['degree'] / math.factorial(r['degree'])
        print(
            f"{r['degree']:>3} {r['n_terms']:>7d} {bound:>14.3e} "
            f"{r['abs_mean_max']:>12.3e} "
            f"{r['std_max']:>12.3e} {r['std_median']:>12.3e} {r['std_min']:>12.3e} "
            f"{r['n_dead_1e8']:>8d} {r['n_dead_10eps']:>10d}"
        )


def _L_stats(L_per_path: torch.Tensor) -> Dict[str, float]:
    return {
        'min': L_per_path.min().item(),
        'median': L_per_path.median().item(),
        'max': L_per_path.max().item(),
    }


def main():
    print(
        f"Poisson process: lambda={LAMBDA}, T={T_MAX}, N={N_SEQS}, "
        f"max_degree={MAX_DEGREE}, seed={SEED}\n"
        f"Path channels: (t, N(t))  ->  paths_dim = {PATH_DIM}\n"
        f"Detector threshold (codebase): std < {DETECTOR_THR:.0e}  "
        f"(float32 eps = {FLOAT32_EPS:.2e}, float64 eps = {FLOAT64_EPS:.2e})\n"
        f"`n<10*eps` shows what a dtype-aware threshold (10 * dtype_eps) would flag."
    )

    paths64, eff_lens = simulate_poisson_paths(N_SEQS, T_MAX, LAMBDA, torch.float64, seed=SEED)
    paths32 = paths64.to(torch.float32)

    # Raw regime: each path's own 1-variation.
    L64 = _L_stats(total_variation_per_path(paths64))
    L32 = _L_stats(total_variation_per_path(paths32))

    rows64, sig64_raw = summarise(paths64, MAX_DEGREE, 10 * FLOAT64_EPS)
    rows32, sig32_raw = summarise(paths32, MAX_DEGREE, 10 * FLOAT32_EPS)

    print_table("RAW float64", rows64, FLOAT64_EPS, L64)
    print_table("RAW float32", rows32, FLOAT32_EPS, L32)

    # Standardised regime: faithful end-to-end pipeline in each dtype.
    paths_std64, L_std64 = standardise_in_dtype(paths64, eff_lens, torch.float64)
    paths_std32, L_std32 = standardise_in_dtype(paths64, eff_lens, torch.float32)
    rows_std64, _ = summarise(paths_std64, MAX_DEGREE, 10 * FLOAT64_EPS)
    rows_std32, _ = summarise(paths_std32, MAX_DEGREE, 10 * FLOAT32_EPS)

    print_table("STANDARDISED float64 (full pipeline in f64)", rows_std64, FLOAT64_EPS, _L_stats(L_std64))
    print_table("STANDARDISED float32 (full pipeline in f32)", rows_std32, FLOAT32_EPS, _L_stats(L_std32))

    # Float64 vs float32 std agreement on raw signatures. Compute once each.
    # Mask near-zero f64 std to avoid the "noise-divided-by-noise" max ratio blowing up.
    NEAR_ZERO = 10 * FLOAT64_EPS
    print(
        "\n=== float64 vs float32 std agreement on RAW signatures "
        f"(coords with f64 std < {NEAR_ZERO:.1e} excluded as numerical-zero) ==="
    )
    for d in range(1, MAX_DEGREE + 1):
        s64 = per_degree_slice(sig64_raw, d, PATH_DIM).std(0)
        s32 = per_degree_slice(sig32_raw, d, PATH_DIM).std(0).double()
        usable = s64.abs() > NEAR_ZERO
        if usable.sum().item() == 0:
            print(f"  degree {d:>2}: all coords below numerical-zero in f64 (skipped)")
            continue
        rel = (s64[usable] - s32[usable]).abs() / s64[usable].abs()
        print(
            f"  degree {d:>2}: median rel-diff = {rel.median().item():.3e}, "
            f"max = {rel.max().item():.3e}  ({int(usable.sum().item())}/{s64.numel()} coords)"
        )


main()
