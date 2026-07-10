import numpy as np
import pytest
from src.generators.noisegenerator import NoiseGenType, get_noise_gen


class TestNoiseGenTypeShapes:
    """Every enum member should produce the correct output shape."""

    @pytest.fixture(params=list(NoiseGenType))
    def noise_type(self, request):
        return request.param

    def test_output_shape(self, noise_type):
        result = noise_type.generate(batch_size=5, length_seqs=10)
        assert result.shape == (5, 10, 1)


class TestNoiseGenTypeStatistics:
    def test_normal_mean_and_std(self):
        """NORMAL samples should have mean near 0 and std near 1 for large N."""
        result = NoiseGenType.NORMAL.generate(batch_size=10000, length_seqs=1)
        assert abs(result.mean()) < 0.1
        assert abs(result.std() - 1.0) < 0.1

    def test_exp_strictly_positive(self):
        """EXP (Poisson inter-arrivals) should produce strictly positive values."""
        result = NoiseGenType.EXP.generate(batch_size=1000, length_seqs=5)
        assert np.all(result > 0)

    def test_exp_two_larger_mean_than_exp(self):
        """EXP_TWO (rate=2) should have smaller mean inter-arrivals than EXP (rate=1)."""
        np.random.seed(42)
        mean_exp = NoiseGenType.EXP.generate(batch_size=5000, length_seqs=1).mean()
        mean_exp2 = NoiseGenType.EXP_TWO.generate(batch_size=5000, length_seqs=1).mean()
        # rate=2 means shorter intervals on average
        assert mean_exp2 < mean_exp

    def test_normal_large_var_has_larger_spread(self):
        np.random.seed(42)
        std_normal = NoiseGenType.NORMAL.generate(batch_size=5000, length_seqs=1).std()
        std_large = NoiseGenType.NORMAL_LARGE_VAR.generate(batch_size=5000, length_seqs=1).std()
        assert std_large > std_normal

    def test_list_all_names(self):
        names = NoiseGenType.list_all_names()
        assert 'normal' in names
        assert 'exp' in names
        assert len(names) == len(NoiseGenType)


class TestGetNoiseGen:
    def test_case_insensitive_lookup(self):
        gen_lower = get_noise_gen("normal")
        gen_upper = get_noise_gen("NORMAL")
        gen_mixed = get_noise_gen("Normal")
        # All should produce the same shape
        assert gen_lower(2, 3).shape == gen_upper(2, 3).shape == gen_mixed(2, 3).shape

    def test_invalid_name_raises(self):
        with pytest.raises(NotImplementedError):
            get_noise_gen("nonexistent_noise_type")

    def test_all_names_resolvable(self):
        """Every name from list_all_names should be resolvable via get_noise_gen."""
        for name in NoiseGenType.list_all_names():
            gen = get_noise_gen(name)
            result = gen(2, 3)
            assert result.shape == (2, 3, 1)
