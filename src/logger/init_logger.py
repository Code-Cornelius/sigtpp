import json
import logging.config
import time

from config import ROOT_DIR
from src.logger.data_log_record import DataLogRecord
from src.utils.utils_os import factory_fct_linked_path

path_linker = factory_fct_linked_path(ROOT_DIR, 'src/logger')


def set_config_logging():
    """
    Initialize the logger with the proper configuration.
    Needs to be called before calling any loggers (including imports) because it disables all loggers previously set.
    We use the colorful handler which outputs colorful messages. We also use the RelativePathFormatter such that
    the path to the file is relative to the root directory (root_dir.py) of the project.
    Finally, we use the DataLogRecord class to lazily log the data in a predefined format.
    """
    logging_config = path_linker(['config_logging.json'])
    try:
        with open(logging_config, 'r') as f:
            config = json.load(f)
            logging.config.dictConfig(config)
            logging.setLogRecordFactory(DataLogRecord)
    except FileNotFoundError:
        print(f"///!\\\\\  Failed to load configuration file {logging_config}")
        # To ensure that this message appears first in the console, we make the threads wait.
        time.sleep(0.2)
        logging.basicConfig(level='INFO')
