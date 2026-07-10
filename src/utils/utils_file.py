import logging
import os
import shutil

logger = logging.getLogger(__name__)


def remove_files_from_dir(folder_path: str, file_start: str = "", file_extension: str = "") -> None:
    """
    Remove all files from a folder with a certain beginning or ending in their name.

    Args:
        folder_path (str): The path to the folder.
        file_start (str): The beginning of the name of the files to be deleted.
        file_extension (str): The ending of the name of the files to be deleted.

    Returns:
        None
    """
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            if file.startswith(file_start) and file.endswith(file_extension):
                file_path = os.path.join(folder_path, file)
                try:
                    os.remove(file_path)
                    logger.info("Removed file: %s", file_path)
                except Exception as e:
                    logger.error("Error removing file %s: %s", file_path, e)
    else:
        logger.warning("Folder %s does not exist. Not deleting anything.", folder_path)
    return


def makedir(directory_where_to_save):
    if not os.path.exists(directory_where_to_save):
        if directory_where_to_save != "":
            os.makedirs(directory_where_to_save)
    return


def delete_dir_tree_safe(path: str, must_be_under: str) -> int:
    """Recursively delete ``path`` and return the number of bytes freed.

    The deletion happens only when ``path`` resolves strictly inside
    ``must_be_under`` (and is not the root itself). Returns 0 without touching
    the filesystem when ``path`` does not exist or fails that containment guard,
    so a malformed name can never escape the intended root.
    """
    if not os.path.isdir(path):
        return 0
    abs_path = os.path.realpath(path)
    abs_root = os.path.realpath(must_be_under)
    if abs_path == abs_root or not abs_path.startswith(abs_root + os.sep):
        print(f"Refusing to delete {path!r}: not strictly under {must_be_under!r}.")
        return 0
    total = 0
    for current_dir, _subdirs, files in os.walk(abs_path):
        for fname in files:
            try:
                total += os.path.getsize(os.path.join(current_dir, fname))
            except OSError:
                pass
    shutil.rmtree(abs_path)
    return total
