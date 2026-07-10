"""Experiment factories for the EDITPP 'yelp_mississauga' dataset."""

from test.paper_experiments.data.real.editpp.yelp_mississauga_dataset import YelpMississaugaDataModule
from test.paper_experiments.settings.editpp_shared import make_editpp_factories

make_editpp_factories("yelp_mississauga", YelpMississaugaDataModule)
