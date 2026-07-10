import numpy as np
import ot
import torch

from src.metrics.tppnorm import pairwise_tpp_area_distance


def auto_reg_from_median(C: np.ndarray, blur_frac: float = 0.05) -> float:
    """
    Suggest an entropic regularization parameter ε for Sinkhorn,
    scaled to the median of the cost matrix entries.
    """
    # 1. Select only strictly positive costs (ignore zeros on the diagonal).
    # In cost matrices, diagonals or matching entries may be zero (distance between identical points).
    # Including zeros would collapse the median. So we filter them out.
    Cpos = C[C > 0]

    # 2. Compute the median cost if there are positive entries,
    #    otherwise fall back to scale = 1.0.
    s = np.median(Cpos) if Cpos.size else 1.0

    # 3. Return ε = blur_frac * scale,
    #    but never smaller than the machine's minimum positive float.
    return max(blur_frac * s, np.finfo(C.dtype).tiny)


@torch.no_grad()
def w1_between_processes_via_tpp_norm(
    X: torch.Tensor,  # (Nf, L, D) fake (assumed aligned with Y)
    Y: torch.Tensor,  # (Nt, L, D) true (assumed aligned with X)
    T,
    reg: float = None,  # None → exact EMD; >0 → Sinkhorn with this reg
    *,
    chunk_size: int = 512,
    numItermax: int = 1_000_000,
    stopThr: float = 1e-6,
    verbose: bool = False,
    method: str = "sinkhorn_log",
) -> float:
    """
    W1 via POT with chunked pairwise cost matrix.
    Exact if reg is None (EMD). Entropic OT if reg>0 (Sinkhorn).
    """
    assert X.ndim == 3 and Y.ndim == 3, f"X,Y must be 3D; got {X.shape},{Y.shape}"
    assert X.shape[1:] == Y.shape[1:], f"(L,D) mismatch: {X.shape[1:]} vs {Y.shape[1:]}"
    assert X.shape[0] > 0 and Y.shape[0] > 0, "Need non-empty batches."

    # Build cost in float64 for stability
    C = pairwise_tpp_area_distance(X, Y, T, chunk_size=chunk_size).cpu().numpy()
    a = np.full(C.shape[0], 1.0 / C.shape[0], dtype=np.float64)
    b = np.full(C.shape[1], 1.0 / C.shape[1], dtype=np.float64)
    return float(
        ot.emd2(a, b, C, numItermax=numItermax)
        if reg is None
        else ot.sinkhorn2(
            a,
            b,
            C,
            auto_reg_from_median(C, float(reg)),
            method=method,
            numItermax=numItermax,
            stopThr=stopThr,
            verbose=verbose,
        )
    ) / (T * T)
