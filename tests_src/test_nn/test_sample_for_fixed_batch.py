"""Tests for TPPArchitecture.sample_for_a_fixed_batch_and_fix and _sample_with_retry.

Strategy
--------
sample_and_fix_seqs is the only stochastic/model-dependent call inside the method
under test.  We mock it on each stub instance and control exactly what it returns
(or whether it raises).  Everything else: cum_times_to_log_inter_times, scaler_exp,
set_seq_to_nan_from_index, concat_two_samples_together, get_num_needed_resample :
runs for real, so the tests also exercise that plumbing.

Mock contract for sample_and_fix_seqs  ->  UnconditionalSamplingResult
    its_scaled_cst   (n_survivors, L1, D)   <- FILTERED scaled ITs, constant-padded
    cum_abs_cst      (n_survivors, L1, D)   <- FILTERED cumulative absolute times
    its_scaled_nan   (n_survivors, L1, D)   <- NaN-masked scaled samples
    cum_rel_nan      (n_survivors, L1, D)   <- NaN-masked cumulative relative times
    its_scaled_raw   (n_input,     L1, D)   <- unfiltered scaled ITs
    cond_its_scaled  (n_input,     L1, D)   <- conditioning scaled ITs (unfiltered)

sample_for_a_fixed_batch_and_fix  ->  ConditionalSamplingResult
    its_scaled_cst   (N, L1, D) or (S, N, L1, D)  <- FILTERED scaled ITs, constant-padded
    cum_abs_cst      (N, L1, D) or (S, N, L1, D)  <- FILTERED cumulative absolute times
    ref_its_nan      (N, L1, D) or (S, N, L1, D)  <- unscaled real ITs, NaN-masked
    gen_its_tf_nan   (N, L1, D) or (S, N, L1, D)  <- unscaled gen ITs (TF), NaN-masked

Dimensions used throughout
    N  = 2   batch size
    L  = 4   length from cum_times_to_log_inter_times  (cumulative -> diff)
    L1 = 3   length after sample_and_fix_seqs strips the first value  (L - 1)
    D  = 1   feature dimension
    S  = 3   num_samples_per_seq for multi-sample tests
"""

import pytest

pytest.importorskip("signatory")  # skips entire module in CI where signatory cannot be built

import unittest
from unittest.mock import MagicMock, patch

import torch


from src.data_types.samplingresult import UnconditionalSamplingResult
from src.nn.architectures.tpp_architecture import TPPArchitecture

_PATCH_TARGET = 'src.nn.architectures.tpp_architecture.tpp_utils.cum_times_to_log_inter_times'

# ---------------------------------------------------------------------------
# Stub subclass :  all abstract methods are no-ops so we can allocate an
# instance without triggering the heavy TPPArchitecture / LightningModule __init__.
# ---------------------------------------------------------------------------


class _StubTPP(TPPArchitecture):
    @staticmethod
    def filter_patho_seqs(tensor1, lens_for_masking, tensor2=None):
        return tensor1, lens_for_masking, tensor2

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def sample(self, *, num_seq=None, starting_times=None, log_inter_arr_times=None):
        pass


# ---------------------------------------------------------------------------
# Canonical dimensions
# ---------------------------------------------------------------------------
N = 2
L = 4
L1 = L - 1  # 3
D = 1
S = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stub():
    """Allocate a _StubTPP without calling __init__; wire only what the method needs.

    cum_times_to_log_inter_times is called as a module-level function inside
    sample_for_a_fixed_batch_and_fix, so tests that exercise that method must
    patch it via _PATCH_TARGET rather than setting an instance attribute.
    """
    obj = object.__new__(_StubTPP)
    # scaler_exp.unscale is identity: we only care about shapes, not values.
    obj.scaler_exp = MagicMock()
    obj.scaler_exp.unscale = lambda x: x
    return obj


def _batch_history():
    """Minimal canonical (data, lens, marks) batch for sample_for_a_fixed_batch_and_fix."""
    times = torch.ones(N, L + 1, D)
    lens = torch.tensor([4, 5])
    marks = torch.zeros(N, L + 1, dtype=torch.long)
    return (times, lens, marks)


def _valid_result(n_survivors, n_input):
    """Build a fake UnconditionalSamplingResult as returned by sample_and_fix_seqs."""
    filt = torch.ones(n_survivors, L1, D)
    unfilt = torch.ones(n_input, L1, D)
    return UnconditionalSamplingResult(
        its_scaled_cst=filt,
        cum_abs_cst=filt.clone(),
        its_scaled_nan=filt.clone(),
        cum_rel_nan=filt.clone(),
        its_scaled_raw=unfilt,
        cond_its_scaled=unfilt.clone(),
    )


# ---------------------------------------------------------------------------
# Tests:  _sample_with_retry  (unit)
# ---------------------------------------------------------------------------


class TestSampleWithRetry(unittest.TestCase):
    """Direct unit tests for the _sample_with_retry helper."""

    def test_returns_immediately_on_success(self):
        obj = _make_stub()
        expected = _valid_result(N, N)
        obj.sample_and_fix_seqs = MagicMock(return_value=expected)

        result = obj._sample_with_retry(
            starting_times=torch.ones(N, 1, D),
            log_inter_arr_times=torch.ones(N, L, D),
            name_phase4logger="test",
        )
        self.assertEqual(result.its_scaled_cst.shape, (N, L1, D))
        obj.sample_and_fix_seqs.assert_called_once()

    def test_retries_on_valueerror_then_succeeds(self):
        """ValueError (from _apply_mask) on first two calls; third succeeds."""
        obj = _make_stub()
        calls = [0]

        def side(**kw):
            calls[0] += 1
            if calls[0] < 3:
                raise ValueError("The training was unstable and all sequences were removed.")
            return _valid_result(N, N)

        obj.sample_and_fix_seqs = MagicMock(side_effect=side)
        result = obj._sample_with_retry(
            starting_times=torch.ones(N, 1, D),
            log_inter_arr_times=torch.ones(N, L, D),
            name_phase4logger="test",
        )
        self.assertEqual(result.its_scaled_cst.shape, (N, L1, D))
        self.assertEqual(obj.sample_and_fix_seqs.call_count, 3)

    def test_retries_on_empty_tensor_then_succeeds(self):
        """filter_patho_seqs returns 0 survivors without raising; next call succeeds."""
        obj = _make_stub()
        calls = [0]

        def side(**kw):
            calls[0] += 1
            if calls[0] == 1:
                return _valid_result(0, N)
            return _valid_result(N, N)

        obj.sample_and_fix_seqs = MagicMock(side_effect=side)
        result = obj._sample_with_retry(
            starting_times=torch.ones(N, 1, D),
            log_inter_arr_times=torch.ones(N, L, D),
            name_phase4logger="test",
        )
        self.assertEqual(result.its_scaled_cst.shape, (N, L1, D))
        self.assertEqual(obj.sample_and_fix_seqs.call_count, 2)

    def test_raises_runtime_error_after_max_attempts(self):
        obj = _make_stub()
        obj.sample_and_fix_seqs = MagicMock(side_effect=ValueError("unstable"))
        with self.assertRaises(RuntimeError):
            obj._sample_with_retry(
                starting_times=torch.ones(N, 1, D),
                log_inter_arr_times=torch.ones(N, L, D),
                name_phase4logger="test",
                max_attempts=3,
            )
        self.assertEqual(obj.sample_and_fix_seqs.call_count, 3)

    def test_raises_runtime_error_on_persistent_empty(self):
        """Every call returns 0 survivors (no exception) -> exhausted."""
        obj = _make_stub()
        obj.sample_and_fix_seqs = MagicMock(return_value=_valid_result(0, N))
        with self.assertRaises(RuntimeError):
            obj._sample_with_retry(
                starting_times=torch.ones(N, 1, D),
                log_inter_arr_times=torch.ones(N, L, D),
                name_phase4logger="test",
                max_attempts=4,
            )
        self.assertEqual(obj.sample_and_fix_seqs.call_count, 4)


# ---------------------------------------------------------------------------
# Tests:  num_samples_per_seq == 1  (single-sample path)
# ---------------------------------------------------------------------------


class TestSingleSample(unittest.TestCase):

    def setUp(self):
        self._patcher = patch(_PATCH_TARGET, return_value=(torch.tensor([3, 4]), torch.ones(N, L, D)))
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    # --- happy path -----------------------------------------------------------

    def test_output_shapes(self):
        """All N sequences survive on first try. Verify all 4 output shapes."""
        obj = _make_stub()
        obj.sample_and_fix_seqs = MagicMock(return_value=_valid_result(N, N))

        result = obj.sample_for_a_fixed_batch_and_fix(_batch_history(), num_samples_per_seq=1, name_phase4logger="test")

        self.assertEqual(result.its_scaled_cst.shape, (N, L1, D))
        self.assertEqual(result.cum_abs_cst.shape, (N, L1, D))
        self.assertEqual(result.ref_its_nan.shape, (N, L1, D))
        self.assertEqual(result.gen_its_tf_nan.shape, (N, L1, D))
        obj.sample_and_fix_seqs.assert_called_once()

    def test_partial_survival_shapes(self):
        """Only 1 of N=2 sequences survives filtering. Filtered outputs shrink."""
        obj = _make_stub()
        obj.sample_and_fix_seqs = MagicMock(return_value=_valid_result(1, N))

        result = obj.sample_for_a_fixed_batch_and_fix(_batch_history(), num_samples_per_seq=1, name_phase4logger="test")
        # Filtered outputs: 1 survivor
        self.assertEqual(result.its_scaled_cst.shape, (1, L1, D))
        self.assertEqual(result.cum_abs_cst.shape, (1, L1, D))

    # --- retry ----------------------------------------------------------------

    def test_retry_on_valueerror(self):
        """First call raises ValueError (_apply_mask); second succeeds."""
        obj = _make_stub()
        calls = [0]

        def side(**kw):
            calls[0] += 1
            if calls[0] == 1:
                raise ValueError("The training was unstable and all sequences were removed.")
            return _valid_result(N, N)

        obj.sample_and_fix_seqs = MagicMock(side_effect=side)
        result = obj.sample_for_a_fixed_batch_and_fix(_batch_history(), num_samples_per_seq=1, name_phase4logger="test")
        self.assertEqual(result.its_scaled_cst.shape, (N, L1, D))
        self.assertEqual(obj.sample_and_fix_seqs.call_count, 2)

    def test_retry_on_empty_tensor(self):
        """First call returns 0 survivors without raising; second succeeds."""
        obj = _make_stub()
        calls = [0]

        def side(**kw):
            calls[0] += 1
            if calls[0] == 1:
                return _valid_result(0, N)
            return _valid_result(N, N)

        obj.sample_and_fix_seqs = MagicMock(side_effect=side)
        result = obj.sample_for_a_fixed_batch_and_fix(_batch_history(), num_samples_per_seq=1, name_phase4logger="test")
        self.assertEqual(result.its_scaled_cst.shape, (N, L1, D))
        self.assertEqual(obj.sample_and_fix_seqs.call_count, 2)

    def test_retry_exhausted_raises(self):
        """Every call raises -> RuntimeError after max_attempts."""
        obj = _make_stub()
        obj.sample_and_fix_seqs = MagicMock(side_effect=ValueError("unstable"))
        with self.assertRaises(RuntimeError):
            obj.sample_for_a_fixed_batch_and_fix(_batch_history(), num_samples_per_seq=1, name_phase4logger="test")


# ---------------------------------------------------------------------------
# Tests:  num_samples_per_seq > 1  (multi-sample path)
# ---------------------------------------------------------------------------


class TestMultiSample(unittest.TestCase):
    """sample_for_a_fixed_batch_and_fix with num_samples_per_seq == S == 3.

    exact_num_sampling defaults to False  ->  oversampling_factor = 1.0
        oversampled_num_per_seq  =  int(S * 1.0)  =  3
    """

    def setUp(self):
        self._patcher = patch(_PATCH_TARGET, return_value=(torch.tensor([3, 4]), torch.ones(N, L, D)))
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    # --- happy path (all survive, no resample) --------------------------------

    def test_output_shapes_no_resample(self):
        """All S=3 copies survive for every sequence.  No resample pass needed."""
        obj = _make_stub()
        # Every call: 3 survivors out of 3 input copies.
        obj.sample_and_fix_seqs = MagicMock(return_value=_valid_result(S, S))

        result = obj.sample_for_a_fixed_batch_and_fix(_batch_history(), num_samples_per_seq=S, name_phase4logger="test")

        # Filtered (stacked): (S, N, L1, D)
        self.assertEqual(result.its_scaled_cst.shape, (S, N, L1, D))
        self.assertEqual(result.cum_abs_cst.shape, (S, N, L1, D))
        # MAPE targets reshaped: (S, N, L1, D)
        self.assertEqual(result.ref_its_nan.shape, (S, N, L1, D))
        self.assertEqual(result.gen_its_tf_nan.shape, (S, N, L1, D))
        # One call per batch element, no resample pass.
        self.assertEqual(obj.sample_and_fix_seqs.call_count, N)

    # --- partial survival  ->  resample triggered ------------------------------------

    def test_resample_pass_triggered(self):
        """2 out of 3 survive in first pass.

        get_num_needed_resample math:
            min_survivors        = 2
            p_success            = 2 / 3
            needed_resample      = int((3 - 2) / (2/3) * 1.0) = int(1.5) = 1

        Resample pass calls sample_and_fix_seqs with 1 copy per sequence.
        concat_two_samples_together pads each sequence from 2 -> 3.
        """
        obj = _make_stub()
        calls = [0]

        def side(**kw):
            calls[0] += 1
            if calls[0] <= N:
                # First pass: 2 survivors, 3 unfiltered (oversampled)
                return _valid_result(2, S)
            # Resample pass: 1 input copy, 1 survivor
            return _valid_result(1, 1)

        obj.sample_and_fix_seqs = MagicMock(side_effect=side)

        result = obj.sample_for_a_fixed_batch_and_fix(_batch_history(), num_samples_per_seq=S, name_phase4logger="test")

        self.assertEqual(result.its_scaled_cst.shape, (S, N, L1, D))
        self.assertEqual(result.cum_abs_cst.shape, (S, N, L1, D))
        # N calls in first pass + N calls in resample pass
        self.assertEqual(obj.sample_and_fix_seqs.call_count, 2 * N)

    # --- retry in first pass ---------------------------------------------------

    def test_retry_valueerror_first_pass(self):
        """n=0 first call raises ValueError; retry succeeds; n=1 is fine first try."""
        obj = _make_stub()
        calls = [0]

        def side(**kw):
            calls[0] += 1
            if calls[0] == 1:
                raise ValueError("unstable")
            return _valid_result(S, S)

        obj.sample_and_fix_seqs = MagicMock(side_effect=side)

        result = obj.sample_for_a_fixed_batch_and_fix(_batch_history(), num_samples_per_seq=S, name_phase4logger="test")
        self.assertEqual(result.its_scaled_cst.shape, (S, N, L1, D))
        # n=0: 1 fail + 1 success = 2;  n=1: 1 success  ->  total 3
        self.assertEqual(obj.sample_and_fix_seqs.call_count, N + 1)

    def test_retry_empty_first_pass(self):
        """n=0 first call returns 0 survivors; retry succeeds."""
        obj = _make_stub()
        calls = [0]

        def side(**kw):
            calls[0] += 1
            if calls[0] == 1:
                return _valid_result(0, S)
            return _valid_result(S, S)

        obj.sample_and_fix_seqs = MagicMock(side_effect=side)

        result = obj.sample_for_a_fixed_batch_and_fix(_batch_history(), num_samples_per_seq=S, name_phase4logger="test")
        self.assertEqual(result.its_scaled_cst.shape, (S, N, L1, D))
        self.assertEqual(obj.sample_and_fix_seqs.call_count, N + 1)

    # --- retry in resample pass ------------------------------------------------

    def test_retry_valueerror_resample_pass(self):
        """First pass: 2 survivors -> resample triggered.
        Resample pass first call raises ValueError; retry succeeds.
        """
        obj = _make_stub()
        calls = [0]

        def side(**kw):
            calls[0] += 1
            if calls[0] <= N:
                # First pass: 2 survivors
                return _valid_result(2, S)
            if calls[0] == N + 1:
                # Resample n=0: raises
                raise ValueError("unstable")
            # Resample n=0 retry or n=1: succeeds
            return _valid_result(1, 1)

        obj.sample_and_fix_seqs = MagicMock(side_effect=side)

        result = obj.sample_for_a_fixed_batch_and_fix(_batch_history(), num_samples_per_seq=S, name_phase4logger="test")
        self.assertEqual(result.its_scaled_cst.shape, (S, N, L1, D))
        # N first + 1 fail + N resample successes = 2*N + 1
        self.assertEqual(obj.sample_and_fix_seqs.call_count, 2 * N + 1)

    # --- retry exhausted -------------------------------------------------------

    def test_retry_exhausted_raises(self):
        """Every call raises ValueError -> RuntimeError on first sequence."""
        obj = _make_stub()
        obj.sample_and_fix_seqs = MagicMock(side_effect=ValueError("unstable"))
        with self.assertRaises(RuntimeError):
            obj.sample_for_a_fixed_batch_and_fix(_batch_history(), num_samples_per_seq=S, name_phase4logger="test")


# ---------------------------------------------------------------------------
# Tests:  Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases(unittest.TestCase):
    """Edge cases: batch_size=1, insufficient survivors, exact_num_sampling."""

    def test_batch_size_one_single_sample(self):
        """S=1 with a single sequence in the batch."""
        N_one = 1
        obj = object.__new__(_StubTPP)
        obj.scaler_exp = MagicMock()
        obj.scaler_exp.unscale = lambda x: x
        obj.sample_and_fix_seqs = MagicMock(return_value=_valid_result(N_one, N_one))

        batch = (torch.ones(N_one, L + 1, D), torch.tensor([4]), torch.zeros(N_one, L + 1, dtype=torch.long))
        with patch(_PATCH_TARGET, return_value=(torch.tensor([3]), torch.ones(N_one, L, D))):
            result = obj.sample_for_a_fixed_batch_and_fix(batch, num_samples_per_seq=1, name_phase4logger="test")

        self.assertEqual(result.its_scaled_cst.shape, (N_one, L1, D))
        self.assertEqual(result.ref_its_nan.shape, (N_one, L1, D))
        obj.sample_and_fix_seqs.assert_called_once()

    def test_batch_size_one_multi_sample(self):
        """S=3 with a single sequence in the batch."""
        N_one = 1
        obj = object.__new__(_StubTPP)
        obj.scaler_exp = MagicMock()
        obj.scaler_exp.unscale = lambda x: x
        # Return S survivors from S input copies
        obj.sample_and_fix_seqs = MagicMock(return_value=_valid_result(S, S))

        batch = (torch.ones(N_one, L + 1, D), torch.tensor([4]), torch.zeros(N_one, L + 1, dtype=torch.long))
        with patch(_PATCH_TARGET, return_value=(torch.tensor([3]), torch.ones(N_one, L, D))):
            result = obj.sample_for_a_fixed_batch_and_fix(batch, num_samples_per_seq=S, name_phase4logger="test")

        # Shapes should have N_one=1 in the batch dimension
        self.assertEqual(result.its_scaled_cst.shape, (S, N_one, L1, D))
        self.assertEqual(result.cum_abs_cst.shape, (S, N_one, L1, D))
        self.assertEqual(result.ref_its_nan.shape, (S, N_one, L1, D))
        self.assertEqual(result.gen_its_tf_nan.shape, (S, N_one, L1, D))
        # One call per batch element, no resample
        self.assertEqual(obj.sample_and_fix_seqs.call_count, N_one)

    def test_exact_num_sampling_shapes(self):
        """exact_num_sampling=True uses 1.25x oversampling. Verify shapes with all survivors."""
        obj = _make_stub()
        S_local = 4
        oversampled = int(S_local * 1.25)  # = 5

        # Return all oversampled copies as survivors
        obj.sample_and_fix_seqs = MagicMock(return_value=_valid_result(oversampled, oversampled))

        with patch(_PATCH_TARGET, return_value=(torch.tensor([3, 4]), torch.ones(N, L, D))):
            result = obj.sample_for_a_fixed_batch_and_fix(
                _batch_history(),
                num_samples_per_seq=S_local,
                name_phase4logger="test",
                exact_num_sampling=True,
            )

        # Final shape should be (S_local, N, L1, D), NOT (oversampled, N, L1, D)
        self.assertEqual(result.its_scaled_cst.shape, (S_local, N, L1, D))
        self.assertEqual(result.cum_abs_cst.shape, (S_local, N, L1, D))
        # One call per batch element, no resample needed (all survived)
        self.assertEqual(obj.sample_and_fix_seqs.call_count, N)

    def test_insufficient_survivors_raises_valueerror(self):
        """When survival rate is too low, concat_two_samples_together fails.

        With exact_num_sampling=True, num_samples_per_seq=100:
        - oversampled_num_per_seq = int(100 * 1.25) = 125
        - If only 2 survivors per call:
          - First pass: 2 survivors per sequence
          - get_num_needed_resample computes needed=7656 (but capped by tensor size)
          - Resample pass also gets ~2 survivors
          - concat tries 2 + 2 = 4 to fill target 100 -> fails
        """
        S_large = 100
        oversampled = int(S_large * 1.25)  # 125

        obj = object.__new__(_StubTPP)
        obj.scaler_exp = MagicMock()
        obj.scaler_exp.unscale = lambda x: x

        def always_two_survivors(**kw):
            """Return only 2 survivors regardless of input size."""
            filt = torch.ones(2, L1, D)
            unfilt = torch.ones(oversampled, L1, D)
            return UnconditionalSamplingResult(
                its_scaled_cst=filt,
                cum_abs_cst=filt.clone(),
                its_scaled_nan=filt.clone(),
                cum_rel_nan=filt.clone(),
                its_scaled_raw=unfilt,
                cond_its_scaled=unfilt.clone(),
            )

        obj.sample_and_fix_seqs = MagicMock(side_effect=always_two_survivors)

        with patch(_PATCH_TARGET, return_value=(torch.tensor([3, 4]), torch.ones(N, L, D))):
            with self.assertRaises(ValueError) as ctx:
                obj.sample_for_a_fixed_batch_and_fix(
                    _batch_history(),
                    num_samples_per_seq=S_large,
                    name_phase4logger="test",
                    exact_num_sampling=True,
                )

        self.assertIn("Not enough values available", str(ctx.exception))

    def test_zero_survivors_in_resample_raises_runtime_error(self):
        """If resample pass returns 0 survivors, _sample_with_retry exhausts retries.

        First pass: 2 survivors -> triggers resample.
        Resample pass: every call raises ValueError -> RuntimeError after exhaustion.
        """
        obj = _make_stub()
        calls = [0]

        def side(**kw):
            calls[0] += 1
            if calls[0] <= N:
                # First pass: partial survival to trigger resample
                return _valid_result(2, S)
            # Resample pass: all filtered -> ValueError
            raise ValueError("unstable")

        obj.sample_and_fix_seqs = MagicMock(side_effect=side)

        with patch(_PATCH_TARGET, return_value=(torch.tensor([3, 4]), torch.ones(N, L, D))):
            with self.assertRaises(RuntimeError) as ctx:
                obj.sample_for_a_fixed_batch_and_fix(_batch_history(), num_samples_per_seq=S, name_phase4logger="test")

        self.assertIn("All samples were filtered as pathological", str(ctx.exception))


# ---------------------------------------------------------------------------
# Tests:  marks shape contract  (Critical-1 + Critical-4 regression guard)
# ---------------------------------------------------------------------------


def _batch_history_with_marks():
    """3-tuple batch_history: (times (N, L+1, D), lens (N,), marks (N, L+1) long)."""
    times = torch.ones(N, L + 1, D)
    lens = torch.tensor([4, 5])
    marks = torch.randint(0, 3, (N, L + 1), dtype=torch.long)
    marks[:, 0] = 0  # anchor mark always 0
    return (times, lens, marks)


class TestMarksShapeContract(unittest.TestCase):
    """Verify marks are anchor-stripped before reaching sample_and_fix_seqs.

    Critical-1 regression: batch_history[2] has shape (N, L+1). After stripping
    the anchor with [:, 1:], marks passed to sample_and_fix_seqs must be (N, L).
    Not (N, L+1) — that would include the anchor.
    Not (N, L-1) — that would over-strip.
    """

    def setUp(self):
        self._patcher = patch(_PATCH_TARGET, return_value=(torch.tensor([3, 4]), torch.ones(N, L, D)))
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_marks_anchor_stripped_single_sample_path(self):
        """Single-sample path: sample_and_fix_seqs receives marks of shape (N, L)."""
        obj = _make_stub()
        captured = {}

        def capture(**kw):
            captured['marks'] = kw.get('marks')
            return _valid_result(N, N)

        obj.sample_and_fix_seqs = MagicMock(side_effect=capture)
        obj.sample_for_a_fixed_batch_and_fix(
            _batch_history_with_marks(), num_samples_per_seq=1, name_phase4logger="test"
        )

        self.assertIsNotNone(captured.get('marks'), "marks should not be None when batch has 3 elements")
        self.assertEqual(
            captured['marks'].shape,
            (N, L),
            f"marks passed to sample_and_fix_seqs should be (N, L)={(N, L)}, got {captured['marks'].shape}. "
            "Anchor (col 0) must be stripped — use [:, 1:] not [:, :-1].",
        )

    def test_legacy_2tuple_batch_is_rejected(self):
        """Legacy 2-tuple batches are no longer accepted."""
        obj = _make_stub()
        with self.assertRaisesRegex(ValueError, "canonical 3-tuple"):
            obj.sample_for_a_fixed_batch_and_fix(
                (torch.ones(N, L + 1, D), torch.tensor([4, 5])),
                num_samples_per_seq=1,
                name_phase4logger="test",
            )


if __name__ == "__main__":
    unittest.main()
