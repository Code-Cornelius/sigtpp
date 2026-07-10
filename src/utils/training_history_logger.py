import logging
import typing
from typing import Dict, Iterable, List, Union

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only

logger = logging.getLogger(__name__)


class TrainingHistoryLogger(LightningLoggerBase):
    """
    A versatile logger class designed to:
        - Save the history of training on the heap instead of in a file.
        - Plot the evolution of the losses during training.
        - Store the hyperparameters of the model.

    **Metrics Handling:**
    - The logger stores metric histories in a `Dictionary of Lists` format where each metric has its own list of `epochs` and `values`.
    - When logging from the `validation_step`, it is essential that the metric names contain certain keywords (e.g., 'val', 'validation') for the logger to correctly handle the validation shift.

    **Plotting Considerations:**
    - The logger periodically plots the metrics based on the configured logging intervals.

    **Implementation Notes:**
    - Ensure that when logging validation metrics, the names include keywords like 'val' or 'validation' for proper handling.
    - The logger is designed to be used within the PyTorch Lightning framework, leveraging its logging and visualization capabilities.

    **Data Structure:**
    - The class uses a `Dictionary of Lists` structure to store the history of each metric. This structure includes separate lists for `epochs` and `values`, which allows for direct access to both epochs and metric values, ensuring efficient data handling during logging and plotting.

    **Difficulties:**
    1. We observed two behaviours. Training and validation are alternated in Pytorch Lightning.
    When backpropagation is doing automatically, it is performed after the validation step.
    However, when backpropagation is done manually, the validation is performed before the backpropagation step.
    This makes the losses not match the "epoch", where the "epoch" represents a concept of a state of the parameters.

    2.  The early stoppers are called before the training of the current step (at epoch 48 in the period=50 example) and after validation of the current step happening before the training of the current step.
    So, in order to log the same loss as used for the best model choice, it is better to early stop on validation losses, which reflect the current state. Otherwise, one observes a shift of 1 epoch.
    Ref: https://github.com/Lightning-AI/pytorch-lightning/issues/1464
    """

    VAL_KEYWORDS: List[str] = ["val", "validation"]

    def __init__(
        self,
        metrics: Iterable[str],
        plot_loss_history: bool = False,
        period_logging_pt_lightning: int = 1,
        period_in_logs_plotting: int = 1,
    ) -> None:
        """
        Args:
            metrics (Iterable[str]): Metrics to be logged and plotted. The order in this list determines the color on the graph.
            plot_loss_history (bool): Whether to plot the loss history during training.
            period_logging_pt_lightning (int): Interval of epochs between logging.
            period_in_logs_plotting (int): Interval of logs before replotting the history.
        """
        # In PL, they call validation every so epoch. We synchronise our variable with theirs:
        # check_val_every_n_epoch = period_logging_pt_lightning
        # check_val_every_n_epoch is found in the trainer.
        super().__init__()

        self.hyper_params: Dict[str, typing.Any] = {}
        self.history: Dict[str, Dict[str, List[Union[float, None]]]] = {}
        self.period_logging_pt_lightning: int = period_logging_pt_lightning
        self.period_in_logs_plotting: int = period_in_logs_plotting

        if plot_loss_history:
            self.fig, self.ax = plt.subplots(figsize=(7, 5))
            self.fig.canvas.manager.set_window_title('Loss History')

            if len(metrics) > 20:
                logging.warning(
                    "The number of metrics is too high to have a clear plot. "
                    "The color palette has only 20 colors and only so many metrics will be plotted."
                )
                self.colors = sns.color_palette(palette='tab20')
                # The colors and alphas are used for successive (on top of each other) drawings.
                self.alphas = np.linspace(0.15, 1.0, len(metrics))[::-1]
            elif len(metrics) > 12:
                self.colors = sns.color_palette(palette='tab20')
                self.alphas = np.linspace(0.15, 1.0, len(metrics))[::-1]
            elif len(metrics) > 8:
                self.colors = sns.color_palette(palette='Paired')
                self.alphas = np.linspace(0.3, 1.0, len(metrics))[::-1]
            else:
                self.colors = sns.color_palette(palette='Dark2')
                self.alphas = np.linspace(0.5, 1.0, len(metrics))[::-1]
            self._lines: Dict[str, typing.Any] = {}
        else:
            self.fig, self.ax = None, None

        self.metrics: Iterable[str] = metrics
        # Adding a nan to the metrics with validation inside, see description of the class for more details.
        for name in self.metrics:
            if TrainingHistoryLogger._is_validation_metric(name):
                self.history[name] = {
                    "epochs": [],
                    "values": [np.nan],
                }
            else:
                self.history[name] = {"epochs": [], "values": []}

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, Union[int, float]], step: int) -> None:
        """
        Logs metrics during training.

        Args:
            metrics (Dict[str, Union[int, float]]): Contains at least the metric name, the value, and the epoch number.
            step (int): The current logging step, should match `check_val_every_n_epoch` from the trainer.
        """
        # The trainer from pytorch lightning logs every check_val_every_n_epoch starting from check_val_every_n_epoch -1.
        # So we account for that shift with the + 1.
        if not ((metrics["epoch"] + 1) % self.period_logging_pt_lightning):
            try:
                # fetch all metrics. We use append (complexity amortized O(1)).
                for metric_name, metric_value in metrics.items():
                    # We exclude the metrics that are from the test step.
                    if metric_name != "epoch" and "test" not in metric_name:
                        self.history[metric_name]["epochs"].append(metrics["epoch"] + 1)
                        self.history[metric_name]["values"].append(metric_value)
            except KeyError as e:
                raise AttributeError(
                    f"KeyError found, potentially you have not instantiated " f"the logger with the key '{e.args[0]}'"
                ) from e

            # Check if the number of logs is a multiple of the plotting period, so we only plot the history when enough new data was stored.
            if not ((metrics["epoch"] + 1) // self.period_logging_pt_lightning) % self.period_in_logs_plotting:
                self.plot_history_prediction()

    def log_hyperparams(self, params: Dict[str, typing.Any], *args: typing.Any, **kwargs: typing.Any) -> None:
        """
        Logs hyperparameters of the model.

        Args:
            params (Dict[str, Any]): Dictionary of hyperparameters.
        """
        self.hyper_params = params

    def finalize(self, status: str) -> None:
        # Called by PyTorch Lightning at end of training. Ensures the plot reflects
        # all logged data, not just the last periodic update.
        self.plot_history_prediction()

    def fetch_score(self, metrics: Union[str, Iterable[str]]) -> List[Dict[str, List[Union[float, None]]]]:
        """
        Fetches the score(s) from the history.

        Args:
            metrics (Union[str, Iterable[str]]): The key or keys to fetch the result.

        Returns:
            List[Dict[str, List[Union[float, None]]]]: List of score(s).
        """
        if isinstance(metrics, str):
            return [self._get_history_one_key(metrics)]
        else:
            return [self._get_history_one_key(key) for key in metrics]

    def _get_metrics_to_plot(self) -> List[str]:
        """Return the plotted metric subset without changing what is logged/stored."""
        return [metric for metric in self.metrics if not (metric == "val_epdf" and "train_epdf" not in self.metrics)]

    def plot_history_prediction(self) -> None:
        """Plots the history of the metrics."""
        metrics_to_plot = self._get_metrics_to_plot()
        losses = self.fetch_score(metrics_to_plot)

        if self.fig is not None:
            if not self._lines:
                # First call: create Line2D artists and configure axes once.
                for color, loss_data, metric_name, alpha in zip(self.colors, losses, metrics_to_plot, self.alphas):
                    (line,) = self.ax.plot(
                        loss_data["epochs"],
                        loss_data["values"],
                        color=color,
                        alpha=alpha,
                        linestyle="--" if TrainingHistoryLogger._is_validation_metric(metric_name) else "-",
                        linewidth=2.5,
                        markersize=0.0,
                        label=metric_name,
                    )
                    self._lines[metric_name] = line
                self.ax.set_title("Dynamical Image of History Training")
                self.ax.set_xlabel("Epochs")
                self.ax.set_ylabel("Loss")
                self.ax.set_yscale("log")
                self.ax.legend(loc="best")
            else:
                # Subsequent calls: update existing lines in-place: no artist teardown.
                for loss_data, metric_name in zip(losses, metrics_to_plot):
                    self._lines[metric_name].set_data(loss_data["epochs"], loss_data["values"])
                self.ax.relim()
                self.ax.autoscale_view()

            plt.pause(0.001)

    def _get_history_one_key(self, key: str) -> Dict[str, List[Union[float, None]]]:
        """
        Retrieves history data for a specific key.

        Args:
            key (str): The metric key to retrieve history for.

        Returns:
            Dict[str, List[Union[float, None]]]: The history of the given metric.
        """
        if key in self.history:
            if self._is_validation_metric(key):
                return {
                    "epochs": self.history[key]["epochs"],
                    "values": self.history[key]["values"][:-1],
                }
            return self.history[key]
        else:
            raise KeyError(
                f"The key {key} does not exist in history. "
                f"If key is supposed to exist, has it been passed to the constructor of the logger?"
            )

    @classmethod
    def _is_validation_metric(cls, key: str) -> bool:
        """
        Checks if a metric is a validation metric.

        Args:
            key (str): The metric key to check.

        Returns:
            bool: True if it's a validation metric, False otherwise.
        """
        return any(val_keyword in key for val_keyword in TrainingHistoryLogger.VAL_KEYWORDS)

    @property
    def name(self):
        return "History_Dict_Logger"

    @property
    def version(self):
        return "V1.0"

    @property
    @rank_zero_experiment
    def experiment(self):
        # Return the experiment object associated with this logger.
        pass
