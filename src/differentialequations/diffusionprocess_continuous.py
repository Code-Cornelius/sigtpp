# Code from https://github.com/mingxuan-yi

import logging
import typing
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
from src.differentialequations.sdetype import SDEType

ScoreNetworkType = typing.Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor
]

# 10000.0 is far for a Gaussian, so it is safe to clamp values there.
# It will also help the model to be more stable.
MAX_VALUE_DIFFUSION = 10000.0


class ContinuousDiffusionProcess(nn.Module):
    @staticmethod
    def get_noise_vector(shape: typing.Tuple[int, ...], device: str) -> torch.Tensor:
        return torch.randn(*shape, device=device)

    def __init__(
        self,
        total_steps: int,
        sde_type: SDEType = SDEType.VP,
        sde_info: Optional[Dict[SDEType, Dict[str, float]]] = None,
    ):
        """
        Initialize the ContinuousDiffusionProcess.

        Args:
            total_steps (int): The total number of steps in the diffusion process.
            sde_type (SDEType): The type of SDE to use (VP, subVP, VE).
            sde_info (Dict[SDEType, Dict[str, float]]): A dictionary containing parameters for each SDE type.
        """
        super().__init__()

        if sde_info is None:
            sde_info = {
                SDEType.VP: {"beta_min": torch.tensor(0.1), "beta_max": torch.tensor(4.0)},
                SDEType.SUB_VP: {
                    "beta_min": torch.tensor(0.1),
                    "beta_max": torch.tensor(20.0),
                },
                SDEType.VE: {
                    "sigma_min": torch.tensor(0.01),
                    "sigma_max": torch.tensor(50.0),
                },
            }

        self.num_diffusion_steps = total_steps

        self.dt: torch.Tensor = nn.Parameter(torch.Tensor([1.0 / self.num_diffusion_steps]), requires_grad=False)
        self.sde_type: SDEType = sde_type
        self.sde_params: Dict[str, float] = sde_info[sde_type]
        self.coefficients: Dict[str, float] = self._compute_coefficients()

        self.linspace_diffusion_steps = torch.nn.Parameter(
            torch.linspace(
                1,
                self.num_diffusion_steps,
                self.num_diffusion_steps,
                dtype=torch.long,
            ),
            requires_grad=False,
        )
        return

    def forward_sample(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate forward samples using the SDE.

        Args:
            data (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The progressively noisier data of shape (S, N, L, D).
        """
        trajectory = [x]

        for time_step in self.linspace_diffusion_steps:
            x = self._forward_one_step(x, time_step)
            trajectory.append(x)
        return torch.stack(trajectory, dim=0)

    def backward_sample(
        self,
        noise: torch.Tensor,
        latent_rep_history: torch.Tensor,
        scaled_running_cum_time: torch.Tensor,
        model: ScoreNetworkType,
    ) -> Tuple[torch.Tensor]:
        x_t = noise
        denoised_data = [x_t]
        for time_step in reversed(self.linspace_diffusion_steps):
            pred_score = model(None, latent_rep_history, scaled_running_cum_time, x_t, time_step.view(1))
            x_t = self._backward_one_step(x_t, time_step, pred_score)
            denoised_data.append(x_t)
        return torch.stack(denoised_data, dim=0)

    def _forward_one_step(self, x_prev: torch.Tensor, t: int) -> torch.Tensor:
        """
        Perform one forward step of the SDE.

        Args:
            x_prev (torch.Tensor): The state at the previous timestep.
            t (torch.Tensor): The current timestep (index) as a tensor of shape (1,).

        Returns:
            torch.Tensor: The state at the next timestep.
        """
        drift, diffusion = self._compute_drift_and_diffusion(x_prev, t)
        noise = torch.randn_like(x_prev)
        # Eq. (25) in Score-based generative modelling through SDEs.
        x_t = x_prev + drift * self.dt + diffusion * noise * torch.sqrt(self.dt)
        return x_t

    def _backward_one_step(
        self,
        x_t: torch.Tensor,
        t: int,
        pred_score: torch.Tensor,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """
        Perform one backward step of the SDE.

        Args:
            x_t (torch.Tensor): The state at the current timestep.
            t (torch.Tensor): The current timestep (index) as a tensor of shape (1,).
            pred_score (torch.Tensor): The predicted score from the model.
            clip_denoised (bool): Whether to clip the denoised output.

        Returns:
            torch.Tensor: The state at the previous timestep.
        """
        drift, diffusion = self._compute_drift_and_diffusion(x_t, t)
        noise = torch.randn_like(x_t, device=x_t.device)
        x_prev = x_t - (drift - diffusion * diffusion * pred_score) * self.dt + diffusion * noise * torch.sqrt(self.dt)

        if clip_denoised and x_t.ndim > 2:
            # Change depending on the dataset.
            x_prev = x_prev.clamp(-MAX_VALUE_DIFFUSION, MAX_VALUE_DIFFUSION)

        return x_prev

    def _compute_coefficients(self) -> Dict[str, float]:
        """
        Compute the coefficients for the SDE based on the selected type.

        Returns:
            Dict[str, float]: The coefficients for the SDE process.
        """
        if self.sde_type in {SDEType.VP, SDEType.SUB_VP}:
            return {
                "beta_0": self.sde_params["beta_min"],
                "beta_1": self.sde_params["beta_max"],
            }

        if self.sde_type is SDEType.VE:
            return {
                "sigma_min": self.sde_params["sigma_min"],
                "sigma_max": self.sde_params["sigma_max"],
            }

    def _compute_drift_and_diffusion(
        self, diffused_process: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute the drift and diffusion coefficients for the SDE at a given timestep.

        Args:
            diffused_process (torch.Tensor): The state at time t of shape (batch_size, *).
            t (torch.Tensor): The current timestep (index) as a tensor of shape (1,).

        Returns:
            Tuple[torch.Tensor, float]: The drift (f_t) and diffusion (g_t) coefficients.
        """
        # Normalize timestep to [0, 1]
        normalized_t: torch.Tensor = t / self.num_diffusion_steps

        if self.sde_type in {SDEType.VP, SDEType.SUB_VP}:
            # Above Eq. (32) in Score-based generative modelling through SDEs.
            beta_t = self.coefficients["beta_0"] + normalized_t * (
                self.coefficients["beta_1"] - self.coefficients["beta_0"]
            )
            drift = -0.5 * beta_t * diffused_process

            if self.sde_type is SDEType.VP:
                diffusion = torch.sqrt(beta_t)
            else:
                discount = 1.0 - torch.exp(
                    -2.0 * self.coefficients["beta_0"] * normalized_t
                    - (self.coefficients["beta_1"] - self.coefficients["beta_0"]) * normalized_t**2.0
                )
                diffusion = torch.sqrt(beta_t * discount)

            return drift, diffusion

        elif self.sde_type is SDEType.VE:
            drift = torch.zeros_like(diffused_process)
            sigma_t = (
                self.coefficients["sigma_min"]
                * (self.coefficients["sigma_max"] / self.coefficients["sigma_min"]) ** normalized_t
            )
            diffusion = sigma_t * torch.sqrt(
                2.0 * (torch.log(self.coefficients["sigma_max"]) - torch.log(self.coefficients["sigma_min"]))
            )
            return drift, diffusion

    def _perturbation_kernel(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, float]]:
        """
        Compute the perturbation kernel for the SDE.

        Args:
            x_0 (torch.Tensor): The initial state at time 0.
            t (torch.Tensor): The current timestep as index.

        Returns:
            Tuple[torch.Tensor, Union[torch.Tensor, float]]: The mean and standard deviation for the perturbation kernel.
        """
        normalized_t: torch.Tensor = t / self.num_diffusion_steps  # Normalize timestep to [0, 1]

        if self.sde_type in {SDEType.VP, SDEType.SUB_VP}:
            log_mean_coeff: torch.Tensor = (
                -0.25 * normalized_t * normalized_t * (self.coefficients["beta_1"] - self.coefficients["beta_0"])
                - 0.5 * normalized_t * self.coefficients["beta_0"]
            )
            mean = torch.exp(log_mean_coeff) * x_0

            if self.sde_type is SDEType.VP:
                std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
            elif self.sde_type is SDEType.SUB_VP:
                std = 1.0 - torch.exp(2.0 * log_mean_coeff)

            return mean, std

        elif self.sde_type is SDEType.VE:
            mean = x_0
            std = (
                self.coefficients["sigma_min"]
                * (self.coefficients["sigma_max"] / self.coefficients["sigma_min"]) ** normalized_t
            )
            return mean, std
