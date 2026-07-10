import matplotlib

matplotlib.use("Agg")

import pytest
import torch
from matplotlib import pyplot as plt

from src.diagnostics import dataset_diagnostics as dd
from src.diagnostics import dataset_plots as dp
from src.diagnostics.dataset_summary import _mark_frequency_stats


class DummyDM:
    def __init__(self):
        self.DATASET_NAME = "dummy"
        self.num_marks = 3
        self.time_max = 30.0

        self.train_in = torch.tensor([[[0.0], [1.0], [2.0]]], dtype=torch.float32)
        self.train_in_len = torch.tensor([3], dtype=torch.long)
        self.train_marks = torch.tensor([[0, 1, 2]], dtype=torch.long)

        self.val_in = torch.tensor([[[0.0], [10.0], [20.0]]], dtype=torch.float32)
        self.val_in_len = torch.tensor([3], dtype=torch.long)
        self.val_marks = torch.tensor([[0, 2, 1]], dtype=torch.long)

        self.test_in = self.val_in.clone()
        self.test_in_len = self.val_in_len.clone()
        self.test_marks = self.val_marks.clone()


def test_event_times_naned_keeps_valid_events_and_masks_padding():
    # set_seq_to_nan_from_index masks positions *strictly after* the given index.
    # For lens=2 (anchor + 1 real event), last valid position in cum[:,1:,:] is 0,
    # so lens-2=0 → mask arange>0 → position 1 (padding) is NaN'd, position 0 is kept.
    cum = torch.tensor([[[0.0], [5.0], [0.0]]], dtype=torch.float32)  # anchor + 1 real + 1 padding cell
    lens = torch.tensor([2], dtype=torch.long)

    result = dd._features._event_times_naned(cum, lens)

    assert result.shape == (1, 2, 1)
    assert torch.isfinite(result[0, 0, 0]), "single valid event should not be masked"
    assert result[0, 0, 0].item() == pytest.approx(5.0)
    assert torch.isnan(result[0, 1, 0]), "padding cell should be NaN"


def test_plot_sequence_length_histogram_tolerates_empty_test_split():
    dm = DummyDM()
    dm.test_in = torch.zeros((0, 3, 1), dtype=torch.float32)
    dm.test_in_len = torch.zeros(0, dtype=torch.long)
    dm.test_marks = torch.zeros((0, 3), dtype=torch.long)

    fig = dp.plot_sequence_length_histogram(dm)
    plt.close(fig)


def test_mark_frequency_stats_excludes_synthetic_anchor():
    dm = DummyDM()

    stats = _mark_frequency_stats(dm.train_marks, dm.train_in_len, dm.num_marks)

    assert stats["total_events"] == 2
    assert stats["counts"] == [0, 1, 1]
    assert stats["frequencies"] == pytest.approx([0.0, 0.5, 0.5])


def test_plot_sequence_marks_uses_requested_split():
    dm = DummyDM()

    fig = dp.plot_sequence_marks(dm, [0], mode="raster", split="val")

    try:
        offsets = fig.axes[0].collections[0].get_offsets()
        assert offsets[:, 0].tolist() == pytest.approx([10.0, 20.0])
        assert "(val)" in fig.axes[0].get_title()
    finally:
        plt.close(fig)


def test_plot_sequence_length_histogram_has_all_split_labels():
    dm = DummyDM()

    fig = dp.plot_sequence_length_histogram(dm)

    try:
        legend = fig.axes[0].get_legend()
        labels = [text.get_text() for text in legend.get_texts()]
        assert "Train" in labels
        assert "Validation" in labels
        assert "Test" in labels
    finally:
        plt.close(fig)


def test_correlation_heatmap_lag_limit_caps_only_when_needed():
    assert dp._correlation_heatmap_lag_limit(12) == 12
    assert dp._correlation_heatmap_lag_limit(40) == 30


def test_shared_effective_lag_caps_acf_display_to_ten():
    series_by_split = {
        "train": torch.zeros((2, 25, 1)),
        "val": torch.zeros((2, 18, 1)),
        "test": torch.zeros((2, 15, 1)),
    }

    assert dp._shared_effective_lag(series_by_split, max_lag=40) == 10


def test_export_dataset_report_raises_after_collecting_panel_failures(monkeypatch, tmp_path):
    dm = DummyDM()

    def ok_panel(*_args, **_kwargs):
        return plt.figure()

    def fail_panel(*_args, **_kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(dp, "plot_sample_paths_comparison", fail_panel)
    monkeypatch.setattr(dp, "plot_intensity_and_its", ok_panel)
    monkeypatch.setattr(dp, "plot_correlation_heatmaps", ok_panel)
    monkeypatch.setattr(dp, "plot_autocorrelation_inter_arrivals_comparison", ok_panel)
    monkeypatch.setattr(dp, "plot_autocorrelation_cumulative_comparison", ok_panel)

    with pytest.raises(RuntimeError, match="sample_paths"):
        dd.export_dataset_report(dm, tmp_path, fig_format="svg")


def test_export_dataset_report_only_renders_paper_panels(monkeypatch, tmp_path):
    dm = DummyDM()
    called = []

    def record_panel(name):
        def _panel(*_args, **_kwargs):
            called.append(name)
            return plt.figure()

        return _panel

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("non-paper panel should not be rendered")

    monkeypatch.setattr(dp, "plot_sample_paths_comparison", record_panel("sample_paths"))
    monkeypatch.setattr(dp, "plot_intensity_and_its", record_panel("intensity_and_its"))
    monkeypatch.setattr(dp, "plot_correlation_heatmaps", record_panel("correlation_heatmap"))
    monkeypatch.setattr(dp, "plot_autocorrelation_inter_arrivals_comparison", record_panel("acf_inter_arrivals"))
    monkeypatch.setattr(dp, "plot_autocorrelation_cumulative_comparison", record_panel("acf_cumulative"))
    monkeypatch.setattr(dp, "plot_sequence_length_histogram", fail_if_called)
    monkeypatch.setattr(dp, "plot_inter_arrival_train_vs_val", fail_if_called)
    monkeypatch.setattr(dp, "plot_mark_frequencies_comparison", fail_if_called)
    monkeypatch.setattr(dp, "plot_sequence_marks_comparison", fail_if_called)

    dd.export_dataset_report(dm, tmp_path, fig_format="svg")

    assert called == [
        "sample_paths",
        "intensity_and_its",
        "correlation_heatmap",
        "acf_inter_arrivals",
        "acf_cumulative",
    ]


def test_export_dataset_report_preview_shows_before_saving(monkeypatch, tmp_path):
    dm = DummyDM()
    events = []

    def ok_panel(*_args, **_kwargs):
        return plt.figure()

    def fake_show():
        events.append("show")

    def fake_savefig(self, path, *args, **kwargs):
        events.append(path.name)

    monkeypatch.setattr(dp, "plot_sample_paths_comparison", ok_panel)
    monkeypatch.setattr(dp, "plot_intensity_and_its", ok_panel)
    monkeypatch.setattr(dp, "plot_correlation_heatmaps", ok_panel)
    monkeypatch.setattr(dp, "plot_autocorrelation_inter_arrivals_comparison", ok_panel)
    monkeypatch.setattr(dp, "plot_autocorrelation_cumulative_comparison", ok_panel)
    monkeypatch.setattr(dd.plt, "show", fake_show)
    monkeypatch.setattr(type(plt.figure()), "savefig", fake_savefig)
    plt.close("all")

    dd.export_dataset_report(dm, tmp_path, fig_format="svg", preview=True)

    assert events[0] == "show"
    assert "dummy_sample_paths.svg" in events[1:]


def test_resolve_dataset_report_dir_defaults_from_datamodule(monkeypatch, tmp_path):
    dm = DummyDM()

    def fake_linker(parts):
        return str(tmp_path.joinpath(*parts))

    monkeypatch.setattr(dd, "_DATASET_DIAGNOSTICS_LINKER", fake_linker)

    assert dd.resolve_dataset_report_dir(dm) == tmp_path / "dummy"
    assert dd.resolve_dataset_report_dir(dm, preview=True) == tmp_path / "dummy_preview"


def test_export_dataset_report_raises_after_collecting_save_failures(monkeypatch, tmp_path):
    dm = DummyDM()
    saved = []

    def ok_panel(*_args, **_kwargs):
        return plt.figure()

    def fake_savefig(self, path, *args, **kwargs):
        if path.name == "dummy_sample_paths.svg":
            raise OSError("disk full")
        saved.append(path.name)

    monkeypatch.setattr(dp, "plot_sample_paths_comparison", ok_panel)
    monkeypatch.setattr(dp, "plot_intensity_and_its", ok_panel)
    monkeypatch.setattr(dp, "plot_correlation_heatmaps", ok_panel)
    monkeypatch.setattr(dp, "plot_autocorrelation_inter_arrivals_comparison", ok_panel)
    monkeypatch.setattr(dp, "plot_autocorrelation_cumulative_comparison", ok_panel)
    monkeypatch.setattr(type(plt.figure()), "savefig", fake_savefig)
    plt.close("all")

    with pytest.raises(RuntimeError, match="sample_paths"):
        dd.export_dataset_report(dm, tmp_path, fig_format="svg")

    assert "dummy_intensity_and_its.svg" in saved
