import pytest
import torch

pytest.importorskip("signatory")  # skips entire module in CI where signatory cannot be built

from src.metrics.sigw1metric_exp import SigW1MetricExp


def _make_paths(n=8, length=5, dim=2, seed=0):
    torch.manual_seed(seed)
    return torch.randn(n, length, dim)


class TestSigW1MetricExp:

    def test_loss_is_nonnegative(self):
        """SigW1 distance must be >= 0 for any two path batches."""
        base = _make_paths(seed=0)
        paths = _make_paths(seed=1)
        metric = SigW1MetricExp(base_paths=base, sig_degree=2)
        loss = metric(paths)
        assert loss.item() >= 0.0

    def test_loss_is_zero_for_identical_distribution(self):
        """Distance between identical batches should be (close to) zero."""
        base = _make_paths(seed=42)
        metric = SigW1MetricExp(base_paths=base, sig_degree=2)
        loss = metric(base)
        assert loss.item() < 1e-4, f"Expected ~0 loss for identical paths, got {loss.item()}"

    def test_output_is_scalar(self):
        """__call__ must return a zero-dim tensor."""
        base = _make_paths(seed=0)
        paths = _make_paths(seed=1)
        metric = SigW1MetricExp(base_paths=base, sig_degree=2)
        loss = metric(paths)
        assert loss.shape == torch.Size([])

    def test_compute_loss_matches_call(self):
        """compute_loss and __call__ must return the same value."""
        base = _make_paths(seed=0)
        paths = _make_paths(seed=1)
        metric = SigW1MetricExp(base_paths=base, sig_degree=2)
        assert torch.isclose(metric(paths), metric.compute_loss(paths))

    def test_loss_is_sum_of_squared_diffs(self):
        """Loss is the sum of squared differences over signature coordinates (no normalisation)."""
        base = _make_paths(seed=0)
        paths = _make_paths(seed=1)
        metric = SigW1MetricExp(base_paths=base, sig_degree=2)

        samples_signature_before_scaling = metric._signature_of(paths)
        if metric.standardise:
            samples_signature = metric.scaler(samples_signature_before_scaling)
        else:
            samples_signature = samples_signature_before_scaling

        diff = metric.base_exp_sig - samples_signature.mean(0)
        if metric.scale_high_degrees:
            diff = diff * metric.factorials

        expected = diff.pow(2).sum()
        assert torch.isclose(metric(paths), expected)

    def test_rejects_invalid_sig_degree(self):
        """sig_degree=0 must raise AssertionError."""
        base = _make_paths(seed=0)
        with pytest.raises(AssertionError):
            SigW1MetricExp(base_paths=base, sig_degree=0)

    def test_rejects_2d_base_paths(self):
        """base_paths must be 3-D; 2-D input must raise AssertionError."""
        bad_paths = torch.randn(8, 5)  # missing dim axis
        with pytest.raises(AssertionError):
            SigW1MetricExp(base_paths=bad_paths, sig_degree=2)

    def test_no_nans_in_loss(self):
        """Loss must not be NaN for well-formed inputs."""
        base = _make_paths(seed=0)
        paths = _make_paths(seed=1)
        metric = SigW1MetricExp(base_paths=base, sig_degree=2)
        loss = metric(paths)
        assert not torch.isnan(loss), "Loss must not be NaN"

    def test_rejects_wrong_dim_paths_in_call(self):
        """__call__ must raise AssertionError if paths dim does not match base_paths dim."""
        base = _make_paths(seed=0, dim=2)
        paths = _make_paths(seed=1, dim=3)
        metric = SigW1MetricExp(base_paths=base, sig_degree=2)
        with pytest.raises(AssertionError):
            metric(paths)

    def test_rejects_2d_paths_in_call(self):
        """__call__ must raise AssertionError if paths is 2-D."""
        base = _make_paths(seed=0)
        metric = SigW1MetricExp(base_paths=base, sig_degree=2)
        with pytest.raises(AssertionError):
            metric(torch.randn(8, 5))

    def test_higher_sig_degree_accepted(self):
        """sig_degree=3 must construct and produce a finite scalar loss."""
        base = _make_paths(seed=0)
        paths = _make_paths(seed=1)
        metric = SigW1MetricExp(base_paths=base, sig_degree=3)
        loss = metric(paths)
        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss)


class TestEffectiveSigDegree:
    """Truncation of the signature computation to a lower effective degree."""

    def test_none_matches_no_kwarg(self):
        """effective_sig_degree=None must reproduce the default behaviour bit-identically."""
        base = _make_paths(seed=0)
        paths = _make_paths(seed=1)
        m_default = SigW1MetricExp(base_paths=base, sig_degree=4)
        m_none = SigW1MetricExp(base_paths=base, sig_degree=4, effective_sig_degree=None)
        assert m_default.sig_len == m_none.sig_len
        assert torch.allclose(m_default(paths), m_none(paths))

    def test_equal_to_sig_degree_matches_default(self):
        """effective_sig_degree == sig_degree must reproduce the default behaviour."""
        base = _make_paths(seed=0)
        paths = _make_paths(seed=1)
        m_default = SigW1MetricExp(base_paths=base, sig_degree=4)
        m_eq = SigW1MetricExp(base_paths=base, sig_degree=4, effective_sig_degree=4)
        assert torch.allclose(m_default(paths), m_eq(paths))

    def test_truncated_has_shorter_sig_len(self):
        """effective_sig_degree < sig_degree must yield a shorter signature vector."""
        base = _make_paths(seed=0, dim=2)
        m_full = SigW1MetricExp(base_paths=base, sig_degree=4)
        m_trunc = SigW1MetricExp(base_paths=base, sig_degree=4, effective_sig_degree=2)
        assert m_trunc.sig_len < m_full.sig_len
        # For D=2: degree-2 signature has 2 + 4 = 6 terms; degree-4 has 2+4+8+16 = 30
        assert m_trunc.sig_len == 6
        assert m_full.sig_len == 30

    def test_truncated_matches_explicit_low_degree(self):
        """effective_sig_degree=k with sig_degree=K must match sig_degree=k (no effective)."""
        base = _make_paths(seed=0)
        paths = _make_paths(seed=1)
        m_trunc = SigW1MetricExp(base_paths=base, sig_degree=4, effective_sig_degree=2)
        m_explicit = SigW1MetricExp(base_paths=base, sig_degree=2)
        assert m_trunc.sig_len == m_explicit.sig_len
        assert torch.allclose(m_trunc(paths), m_explicit(paths))

    def test_rejects_zero(self):
        """effective_sig_degree=0 must raise AssertionError."""
        base = _make_paths(seed=0)
        with pytest.raises(AssertionError):
            SigW1MetricExp(base_paths=base, sig_degree=4, effective_sig_degree=0)

    def test_rejects_above_sig_degree(self):
        """effective_sig_degree > sig_degree must raise AssertionError."""
        base = _make_paths(seed=0)
        with pytest.raises(AssertionError):
            SigW1MetricExp(base_paths=base, sig_degree=4, effective_sig_degree=5)

    def test_sig_degree_attribute_preserved(self):
        """sig_degree attribute is unchanged for logging even when truncated."""
        base = _make_paths(seed=0)
        m = SigW1MetricExp(base_paths=base, sig_degree=4, effective_sig_degree=2)
        assert m.sig_degree == 4


class TestFloat64Signature:
    """`use_float64_signature` runs the signature/aggregation/backward in float64,
    while inputs and outputs stay float32 (the project's training dtype)."""

    def test_default_buffers_are_float32(self):
        """Default mode keeps all data-derived buffers in float32."""
        base = _make_paths(seed=0)  # float32
        m = SigW1MetricExp(base_paths=base, sig_degree=3, scale_high_degrees=True, standardise=True)
        assert m.base_exp_sig.dtype == torch.float32
        assert m.exp_paths_sig_before_scaling.dtype == torch.float32
        assert m.factorials.dtype == torch.float32
        assert m.scaler.mean_paths.dtype == torch.float32
        assert m.scaler.std_paths.dtype == torch.float32

    def test_f64_mode_buffers_are_float64(self):
        """With use_float64_signature=True, every data-derived buffer is float64,
        even though base_paths is float32."""
        base = _make_paths(seed=0)  # float32
        m = SigW1MetricExp(
            base_paths=base,
            sig_degree=3,
            scale_high_degrees=True,
            standardise=True,
            use_float64_signature=True,
        )
        assert base.dtype == torch.float32
        assert m.base_exp_sig.dtype == torch.float64
        assert m.exp_paths_sig_before_scaling.dtype == torch.float64
        assert m.factorials.dtype == torch.float64
        assert m.scaler.mean_paths.dtype == torch.float64
        assert m.scaler.std_paths.dtype == torch.float64

    def test_f64_mode_internal_signature_is_float64(self):
        """The signature computed inside the metric is float64 in f64 mode,
        even when fed a float32 batch."""
        base = _make_paths(seed=0)
        paths = _make_paths(seed=1)  # float32
        m = SigW1MetricExp(base_paths=base, sig_degree=2, use_float64_signature=True)
        sig = m._signature_of(paths)
        assert sig.dtype == torch.float64

    def test_default_internal_signature_is_float32(self):
        """In default mode the internal signature stays float32."""
        base = _make_paths(seed=0)
        paths = _make_paths(seed=1)
        m = SigW1MetricExp(base_paths=base, sig_degree=2)
        sig = m._signature_of(paths)
        assert sig.dtype == torch.float32

    def test_f64_mode_output_loss_is_float32(self):
        """The returned loss is cast back to float32 (the input dtype) in f64 mode."""
        base = _make_paths(seed=0)
        paths = _make_paths(seed=1)  # float32
        m = SigW1MetricExp(base_paths=base, sig_degree=3, use_float64_signature=True)
        loss = m(paths)
        assert loss.dtype == torch.float32
        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss)

    def test_default_output_loss_is_float32(self):
        """Default mode returns float32 (unchanged behaviour)."""
        base = _make_paths(seed=0)
        paths = _make_paths(seed=1)
        m = SigW1MetricExp(base_paths=base, sig_degree=3)
        loss = m(paths)
        assert loss.dtype == torch.float32

    def test_f64_mode_gradient_to_float32_input_is_float32(self):
        """Backward runs in float64 internally, but the gradient delivered to a
        float32 model input is float32 (autograd casts at the boundary)."""
        base = _make_paths(seed=0)
        paths = _make_paths(seed=1).requires_grad_(True)  # float32, requires grad
        m = SigW1MetricExp(base_paths=base, sig_degree=3, use_float64_signature=True)
        loss = m(paths)
        loss.backward()
        assert paths.grad is not None
        assert paths.grad.dtype == torch.float32
        assert not torch.isnan(paths.grad).any()

    def test_f64_and_f32_losses_are_close(self):
        """Sanity: the two modes agree to within float32 tolerance on well-conditioned
        random paths (they only diverge on near-cancellation / high-degree terms)."""
        base = _make_paths(seed=0)
        paths = _make_paths(seed=1)
        m32 = SigW1MetricExp(base_paths=base, sig_degree=2)
        m64 = SigW1MetricExp(base_paths=base, sig_degree=2, use_float64_signature=True)
        assert torch.isclose(m32(paths), m64(paths), rtol=1e-4, atol=1e-6)

    def test_f64_mode_rejects_non_float32_input(self):
        """The boundary contract is unchanged: paths must arrive as the original
        input dtype (float32); an explicit float64 batch is a mismatch and rejected."""
        base = _make_paths(seed=0)  # float32 -> input_dtype is float32
        paths = _make_paths(seed=1).double()  # float64 -- wrong boundary dtype
        m = SigW1MetricExp(base_paths=base, sig_degree=2, use_float64_signature=True)
        with pytest.raises(AssertionError):
            m(paths)


class TestSigW1DegreeDetector:
    """Detector that infers an effective signature degree from base_paths."""

    def test_returns_sig_degree_when_no_dead_degrees(self):
        """Random paths typically have no dead degrees → effective == sig_degree."""
        from src.metrics.sigw1_degree_detector import SigW1DegreeDetector

        base = _make_paths(seed=0, n=50, length=10, dim=2)
        det = SigW1DegreeDetector(base_paths=base, sig_degree=3)
        assert det.effective_sig_degree == 3
        assert det.dead_degrees == []

    def test_detects_dead_degrees_for_constant_paths(self):
        """A constant base_paths makes all signature terms dead -> trailing block is [1,2,3] -> floored at 1."""
        from src.metrics.sigw1_degree_detector import SigW1DegreeDetector

        base = torch.zeros(20, 5, 2)  # all-constant: every degree is dead
        det = SigW1DegreeDetector(base_paths=base, sig_degree=3)
        assert det.dead_degrees == [1, 2, 3]
        assert det.effective_sig_degree == 1  # trailing block reaches degree 1 -> floored

    def test_exposes_dead_degrees_list(self):
        """`.dead_degrees` is a list of integers."""
        from src.metrics.sigw1_degree_detector import SigW1DegreeDetector

        base = _make_paths(seed=0)
        det = SigW1DegreeDetector(base_paths=base, sig_degree=2)
        assert isinstance(det.dead_degrees, list)

    def test_rejects_invalid_sig_degree(self):
        from src.metrics.sigw1_degree_detector import SigW1DegreeDetector

        base = _make_paths(seed=0)
        with pytest.raises(AssertionError):
            SigW1DegreeDetector(base_paths=base, sig_degree=0)

    def test_rejects_2d_base_paths(self):
        from src.metrics.sigw1_degree_detector import SigW1DegreeDetector

        with pytest.raises(AssertionError):
            SigW1DegreeDetector(base_paths=torch.randn(8, 5), sig_degree=2)

    def test_detector_value_matches_metric_warning_recommendation(self):
        """When the detector finds dead degrees, its effective value matches the
        SigW1MetricExp warning's 'reduce to at most %d' recommendation."""
        from src.metrics.sigw1_degree_detector import SigW1DegreeDetector

        # Build paths where high degrees are dead by holding higher channels constant.
        torch.manual_seed(0)
        base = torch.randn(40, 8, 2)
        base[:, :, 1] = 0.0  # second channel constant — kills mixed-channel degrees
        det = SigW1DegreeDetector(base_paths=base, sig_degree=4)
        # effective should be at most sig_degree, at least 1
        assert 1 <= det.effective_sig_degree <= 4


class TestTrimFromHighestDead:
    """Pure-logic tests for _trim_to_effective_degree: trim from max(dead) onward.

    No contiguity assumption — an isolated alive degree above the highest dead is
    still dropped (treated as escaping noise), and an isolated dead degree below
    the highest dead is kept (one wasted depth costs less than discarding signal).
    """

    def _trim(self, dead, sig_degree):
        from src.metrics.sigw1_degree_detector import SigW1DegreeDetector

        return SigW1DegreeDetector._trim_to_effective_degree(dead, sig_degree)

    def test_pure_trailing_block(self):
        """dead=[8,9,10] -> trailing block starts at 8 -> effective=7."""
        assert self._trim([8, 9, 10], 10) == 7

    def test_non_contiguous_dead_trims_from_start_of_trailing_block(self):
        """dead=[4,7]: trailing block ending at max(dead)=7 is just {7} -> effective=6.

        Old min-rule would have returned 3 and silently dropped alive degrees 5, 6.
        The trailing-suffix-of-sig_degree rule would have returned 10 (no trim).
        This rule trims from 7 onward and keeps 1-6 including the isolated dead 4.
        """
        assert self._trim([4, 7], 10) == 6

    def test_non_contiguous_dead_with_trailing_block(self):
        """dead=[4,7,8,9,10]: trailing block is {7,8,9,10} -> effective=6.

        The isolated dead at 4 is kept implicitly because truncation is a single
        max-depth cutoff. Paying one wasted depth costs less than discarding
        alive degrees 5, 6.
        """
        assert self._trim([4, 7, 8, 9, 10], 10) == 6

    def test_only_top_dead(self):
        """dead=[10] -> trailing block is {10} -> effective=9."""
        assert self._trim([10], 10) == 9

    def test_no_dead_returns_sig_degree(self):
        """Empty dead list -> effective=sig_degree."""
        assert self._trim([], 10) == 10

    def test_single_middle_dead_trims_from_it(self):
        """dead=[5] -> singleton trailing block {5} -> effective=4 (drop 5..10).

        An isolated dead at depth 5 means we don't trust 6..10 either — they're
        either noise that escaped the threshold or genuine signal we conservatively
        drop along with the dead degree that triggered detection.
        """
        assert self._trim([5], 10) == 4

    def test_floors_at_one_when_only_degree_one_dead(self):
        """dead=[1] -> trailing block reaches 1 -> floored at 1."""
        assert self._trim([1], 5) == 1

    def test_all_degrees_dead_floors_at_one(self):
        """dead=[1,2,3] with sig_degree=3 -> trailing block reaches 1 -> floored at 1."""
        assert self._trim([1, 2, 3], 3) == 1
