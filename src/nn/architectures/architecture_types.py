from enum import Enum
from typing import List, Type

from src.nn.architectures.architecture_deter import ArchitectureDeter
from src.nn.architectures.architecture_gamma import ArchitectureGamma
from src.nn.architectures.architecture_one_to_one import ArchitectureOneToOne
from src.nn.architectures.architecture_ddpm import Architecture_DDPM
from src.nn.architectures.architecture_vae import Architecture_VAE
from src.nn.architectures.architecture_wgan_baseline import Architecture_wgan_baseline
from src.nn.architectures.tpp_architecture import TPPArchitecture


class Architectures(str, Enum):
    DDPM = "ddpm"
    WGAN = "wgan"
    SIGTPP = "sigtpp"
    DETER = "deter"
    GAMMA = "gamma"
    VAE = "vae"

    @staticmethod
    def get_model_class(architecture: 'Architectures') -> Type[TPPArchitecture]:
        model_class_map = {
            Architectures.SIGTPP: ArchitectureOneToOne,
            Architectures.WGAN: Architecture_wgan_baseline,
            Architectures.DDPM: Architecture_DDPM,
            Architectures.DETER: ArchitectureDeter,
            Architectures.GAMMA: ArchitectureGamma,
            Architectures.VAE: Architecture_VAE,
        }
        try:
            return model_class_map[architecture]
        except KeyError:
            raise ValueError(f"Unknown architecture: {architecture}")

    @staticmethod
    def get_metrics(architecture: 'Architectures', num_marks: int) -> List[str]:
        metrics_map = {
            Architectures.SIGTPP: [
                'train_sigW',
                'val_sigW',
                'val_epdf',
                'val_hist_it',
                'val_hist_int',
            ],
            Architectures.WGAN: [
                'train_wasserstein',
                'val_wasserstein',
                'val_epdf',
                'val_hist_it',
                'val_hist_int',
                'train_lip_loss',
            ],
            Architectures.DDPM: [
                'train_score',
                'val_score',
                'val_epdf',
                'val_hist_it',
                'val_hist_int',
            ],
            Architectures.DETER: [
                'train_MSE',
                'val_MSE',
                'val_sigW',
                'val_epdf',
                'val_hist_it',
                'val_hist_int',
            ],
            Architectures.GAMMA: ['train_nll', 'val_sigW', 'val_epdf', 'val_hist_it', 'val_hist_int'],
            Architectures.VAE: [
                'train_elbo',
                'val_elbo',
                'train_kl',
                'val_kl',
                'train_recon',
                'val_recon',
                'val_epdf',
                'val_hist_it',
                'val_hist_int',
            ],
        }
        try:
            base = metrics_map[architecture]
        except KeyError:
            raise ValueError(f"Unknown architecture: {architecture}")
        if num_marks > 1 and architecture in _LEARNABLE_MARK_ARCHITECTURES:
            base = base + ['train_mark_ce', 'val_mark_ce']
        if num_marks > 1:
            base = base + ['val_top1_mark_acc']
        if num_marks >= 3 and architecture in _TOP3_MARK_ACCURACY_ARCHITECTURES:
            base = base + ['val_top3_mark_acc']
        return base

    @classmethod
    def _missing_(cls, value):
        """
        Custom behavior for handling missing enum members during instantiation.

        This method enables case-insensitive and whitespace-trimmed matching
        of enum values, allowing instantiation with variants like:
        - Architectures("SIGTPP")
        - Architectures("  sigtpp ")

        Args:
            value (Any): The value attempted to be used for enum instantiation.

        Returns:
            Optional[Architectures]: A matching enum member if found, else None.

        Note:
            If no matching member is found, Python will raise a ValueError automatically.
        """
        if not isinstance(value, str):
            return None
        normalized = value.strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        return None


# Architectures that support a learnable mark head (CE-trained mark predictor).
_LEARNABLE_MARK_ARCHITECTURES = frozenset(
    {
        Architectures.SIGTPP,
        Architectures.WGAN,
        Architectures.DDPM,
        Architectures.VAE,
    }
)

# Architectures that emit top-3 mark accuracy during validation/test when num_marks >= 3.
_TOP3_MARK_ACCURACY_ARCHITECTURES = frozenset(
    {
        Architectures.GAMMA,
        *_LEARNABLE_MARK_ARCHITECTURES,
    }
)
