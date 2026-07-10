from test.paper_experiments.data.real.editpp.editpp_dataset import EditTPPDataModule
from test.paper_experiments.data.real.editpp.editpp_dataset import preview_editpp_datamodule


class YelpMississaugaDataModule(EditTPPDataModule):
    """Yelp Mississauga check-in events (EDITPP 'yelp_mississauga' dataset)."""

    DATASET_FILE = "yelp_mississauga.pkl"


if __name__ == '__main__':
    preview_editpp_datamodule(YelpMississaugaDataModule)
