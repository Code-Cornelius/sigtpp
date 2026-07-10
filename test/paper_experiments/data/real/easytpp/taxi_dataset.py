from test.paper_experiments.data.real.easytpp.easytpp_dataset import EasyTPPDataModule


class TaxiDataModule(EasyTPPDataModule):
    """NYC taxi pick-up events (EasyTPP 'taxi' dataset)."""

    DATASET_NAME = "taxi"
    # Pinned to easytpp/taxi main HEAD as of 2024-02-14 for reproducibility.
    DATASET_REVISION = "31af59ee1ae184b5b7abbf0094099b1172d66e2a"


if __name__ == "__main__":
    from src.logger.init_logger import set_config_logging

    set_config_logging()
    from src.diagnostics.dataset_diagnostics import export_dataset_report

    data = TaxiDataModule()
    print(f"num_marks={data.num_marks}, time_max={data.time_max:.2f}")
    export_dataset_report(data, fig_format="pdf", preview=False)
