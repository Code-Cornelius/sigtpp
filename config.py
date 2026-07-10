import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

INF_INT: int = sys.maxsize
# Value to add to divisions or to the log of a value to avoid division by zero or log of zero for tpps.
EPSILON_STABILITY = 1e-12

OUT_FILE_NAME = "out"  # if you change this, change also output_dir in the experiments.
