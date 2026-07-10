import os

import colorlog

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class LogRelativePathFormatter(colorlog.ColoredFormatter):
    # Extends ColoredFormatter because we want to be able to use it with ColoredFormatter.
    # We replace the full path with the relative path when using the variable %(pathname)s.
    root_path = ROOT_DIR

    _LEVEL_ABBREV = {"WARNING": "WARN", "CRITICAL": "CRIT"}

    def format(self, record):
        record.pathname = os.path.relpath(record.pathname, self.root_path)
        record.levelname = self._LEVEL_ABBREV.get(record.levelname, record.levelname)
        return super().format(record)
