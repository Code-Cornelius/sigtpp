from test.paper_experiments.data.real.easytpp.easytpp_dataset import EasyTPPDataModule


class EarthquakeDataModule(EasyTPPDataModule):
    """Earthquake occurrence events (EasyTPP 'earthquake' dataset)."""

    DATASET_NAME = "earthquake"
    # Pinned to easytpp/earthquake main HEAD as of 2024-02-14 for reproducibility.
    DATASET_REVISION = "1c16ceff43689ab725fc95d69c58cba4d88af247"


if __name__ == "__main__":
    from src.logger.init_logger import set_config_logging

    set_config_logging()
    from src.diagnostics.dataset_diagnostics import export_dataset_report

    data = EarthquakeDataModule()
    print(f"num_marks={data.num_marks}, time_max={data.time_max:.2f}")
    export_dataset_report(data, fig_format="svg", preview=False, max_paths=50)
