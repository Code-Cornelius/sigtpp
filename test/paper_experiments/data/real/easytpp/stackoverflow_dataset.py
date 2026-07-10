from test.paper_experiments.data.real.easytpp.easytpp_dataset import EasyTPPDataModule


class StackOverflowDataModule(EasyTPPDataModule):
    """User badge reward events (EasyTPP 'stackoverflow' dataset)."""

    DATASET_NAME = "stackoverflow"
    # Pinned to easytpp/stackoverflow main HEAD as of 2024-04-05 for reproducibility.
    DATASET_REVISION = "d356ab0d8bc4dd65f5e1f1f2ca9c0759bcfa7344"


if __name__ == "__main__":
    from src.logger.init_logger import set_config_logging

    set_config_logging()
    from src.diagnostics.dataset_diagnostics import export_dataset_report

    data = StackOverflowDataModule()
    print(f"num_marks={data.num_marks}, time_max={data.time_max:.2f}")
    export_dataset_report(data, fig_format="svg", preview=True, max_paths=50)
