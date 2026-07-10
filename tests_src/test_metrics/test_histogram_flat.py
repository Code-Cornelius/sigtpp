import pytest

pytest.importorskip("signatory")  # TPPMetrics constructs SigW1MetricExp; signatory is unavailable in CI.

import torch

from src.data_types.tppmetrics import TPPMetrics, TPPMetricsConfig


@pytest.fixture
def small_metrics():
    """Minimal deterministic TPPMetrics for shape/value tests."""
    torch.manual_seed(0)
    N, Lp1, D = 32, 6, 1
    cum = torch.cumsum(torch.rand(N, Lp1 - 1, D) + 0.1, dim=1)
    cum = torch.cat([torch.zeros(N, 1, D), cum], dim=1)  # anchor at t=0
    lens = torch.full((N,), Lp1, dtype=torch.long)

    class _IdentityScaler:
        def __call__(self, x):
            return x

        def forward(self, x):
            return x

        def unscale(self, x):
            return x

    return TPPMetrics(
        reference_data=cum,
        reference_lens=lens,
        scaler=_IdentityScaler(),
        config=TPPMetricsConfig(num_bins=10),
        sig_loss_seqs=torch.zeros(N, Lp1 - 2, 2 * D),
        scale_paths_pre_sig=lambda x, seq_lens=None: x,
    )


def test_flat_buffers_initialised(small_metrics):
    assert hasattr(small_metrics, 'histogram_loss_it_flat')
    assert hasattr(small_metrics, 'histogram_loss_cum_flat')


def test_compute_histogram_metrics_flat_keys(small_metrics):
    gen_it = small_metrics.reference_data_naned.clone()
    gen_cum = small_metrics.reference_data_cum_naned.clone()
    out = small_metrics.compute_histogram_metrics_flat(gen_it, gen_cum)
    assert set(out.keys()) == {'hist_it_flat', 'hist_int_flat'}
    for v in out.values():
        assert isinstance(v, float)


def test_hist_flat_zero_when_input_equals_reference(small_metrics):
    gen_it = small_metrics.reference_data_naned.clone()
    gen_cum = small_metrics.reference_data_cum_naned.clone()
    out = small_metrics.compute_histogram_metrics_flat(gen_it, gen_cum)
    assert out['hist_it_flat'] < 1e-3
    assert out['hist_int_flat'] < 1e-3
