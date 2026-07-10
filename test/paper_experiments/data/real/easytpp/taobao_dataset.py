from test.paper_experiments.data.real.easytpp.easytpp_dataset import EasyTPPDataModule


class TaobaoDataModule(EasyTPPDataModule):
    """Online shopping behavior events (EasyTPP 'taobao' dataset)."""

    DATASET_NAME = "taobao"
    # Pinned to easytpp/taobao main HEAD as of 2024-05-26 for reproducibility.
    DATASET_REVISION = "a96ec6dc7ef311ca45993af9c8a80f78ed08ae47"


if __name__ == "__main__":
    from src.logger.init_logger import set_config_logging

    set_config_logging()
    from src.diagnostics.dataset_diagnostics import export_dataset_report

    data = TaobaoDataModule()
    print(f"num_marks={data.num_marks}, time_max={data.time_max:.2f}")
    export_dataset_report(data, fig_format="pdf", preview=False)
