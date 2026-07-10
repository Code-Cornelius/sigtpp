# Code from https://github.com/mingxuan-yi

from enum import Enum


class SDEType(str, Enum):
    """Enum representing the types of Stochastic Differential Equations (SDEs)."""

    VP = "VP"  # Variance Preserving
    SUB_VP = "subVP"  # Sub Variance Preserving
    VE = "VE"  # Variance Exploding
