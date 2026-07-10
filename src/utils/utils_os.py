import logging
import os
from contextlib import contextmanager

from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)

from src.utils.utils_file import makedir


def factory_fct_linked_path(ROOT_DIR, path_to_folder):
    """
    Semantics:

    Args:
        ROOT_DIR: path to the root of the project.
        path_to_folder: a path written in the format you want because we use the function os.path.join to link it.

    Returns:
        The linker
    Examples:
              linked_path = factory_fct_linked_path(ROOT_DIR, "path/a"):
              path_save_history = linked_path(['plots', f"best_score_{nb}.pth"])
              #and ROOT_DIR should be imported from a script at the root where it is written:

              import os
              ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    """
    # Normalize and then replace Windows-style backslashes with forward slashes
    PATH_TO_ROOT = os.path.join(ROOT_DIR, path_to_folder).replace("\\", "/")

    def linked_path(path):
        # a list of folders like: ['C','users','name'...]
        # when adding a '' at the end like
        #       path_to_directory = linker_path_to_result_file([path, ''])
        # one adds a / at the end of the path. This is necessary in order to continue writing the path.
        if path.count("C:") > 1:
            logger.warning(f"Warning: multiple 'C:' detected in constructed path: {path}")
        return os.path.join(PATH_TO_ROOT, *path).replace("\\", "/")

    return linked_path


def rmv_file(file_path):
    """
    Semantics:
        Wrapper around os to remove a file. It will call remove only if they file exists, nothing otherwise.

    Args:
        file_path: The full path to the file.

    Returns:
        Void.
    """
    if os.path.isfile(file_path):
        os.remove(file_path)
    else:
        print(f"File {file_path} does not exist or is not a file. File not removed.")
    return


def savefig(fig: plt.Figure, path_file: str) -> None:
    """
    Saves a matplotlib figure to the specified file path.

    Args:
        fig (plt.Figure): The matplotlib figure to save.
        path_file (str): The full path to the file where the figure should be saved.
                         The path should include the file extension, for example .png but not mandatory (png default).

    More information @https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html

    Returns:
        None
    """
    directory_where_to_save = os.path.dirname(path_file)
    makedir(directory_where_to_save)
    fig.savefig(path_file)
    return


@contextmanager
def suppress_logging(level=logging.WARNING):
    logger = logging.getLogger()
    original_level = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(original_level)
