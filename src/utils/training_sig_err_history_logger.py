from typing import Iterable, Optional

import numpy as np
from matplotlib import pyplot as plt

from src.utils.training_history_logger import TrainingHistoryLogger
from src.utils.utils_os import savefig


class TrainingSigErrHistoryLogger(TrainingHistoryLogger):
    """
    A subclass of TrainingHistoryLogger designed to handle plotting with an additional base error reference.

    Attributes:
        base_error (Optional[float]): A fixed error value to be displayed on the plot as a reference line.
    """

    def __init__(
        self,
        metrics: Iterable[str],
        plot_loss_history: bool = False,
        period_logging_pt_lightning: int = 1,
        period_in_logs_plotting: int = 1,
        base_error: Optional[float] = None,
        output_dir: Optional[str] = None,
    ) -> None:
        """
        Initializes the logger with the option to include a base error line in the plot.

        Args:
            metrics (Iterable[str]): Metrics to be logged and plotted.
            plot_loss_history (bool): Whether to plot the loss history during training.
            period_logging_pt_lightning (int): Interval of epochs between logging.
            period_in_logs_plotting (int): Interval of logs before replotting the history.
            base_error (Optional[float]): Fixed error value to show on the plot as a dashed line (default: None).
        """
        super().__init__(
            metrics=metrics,
            plot_loss_history=plot_loss_history,
            period_logging_pt_lightning=period_logging_pt_lightning,
            period_in_logs_plotting=period_in_logs_plotting,
        )
        self.base_error = base_error
        self.base_error_histo: Optional[float] = None
        self.output_dir = output_dir

    def set_base_error(self, base_train_loss_err: float, base_histo_loss_err: Optional[float] = None) -> None:
        """
        Sets the base error value to be used as a reference line on the plot.

        Args:
            base_train_loss_err (float): The base error value of the training loss.
            base_histo_loss_err (float): The base error value of the histogram loss.
        """
        self.base_error: float = base_train_loss_err
        self.base_error_histo: Optional[float] = base_histo_loss_err
        return

    def finalize(self, status: str) -> None:
        # Called by PyTorch Lightning at end of training. Overrides base to also
        # save the figure if output_dir is set, in addition to refreshing the plot.
        self.plot_history_prediction()

    def plot_history_prediction(self) -> None:
        """
        Plots the history of metrics along with an optional base error reference line.
        """
        metrics_to_plot = self._get_metrics_to_plot()
        losses = self.fetch_score(metrics_to_plot)

        if self.fig is not None:
            if not self._lines:
                # First call: create Line2D artists and configure axes once.
                for color, loss_data, metric_name, alpha in zip(self.colors, losses, metrics_to_plot, self.alphas):
                    epochs = np.array(loss_data["epochs"])[~np.isnan(loss_data["values"])]
                    values = np.array(loss_data["values"])[~np.isnan(loss_data["values"])]
                    (line,) = self.ax.plot(
                        epochs,
                        values,
                        color=color,
                        alpha=alpha,
                        linestyle=("--" if TrainingHistoryLogger._is_validation_metric(metric_name) else "-"),
                        linewidth=2.5,
                        markersize=0.0,
                        label=metric_name,
                    )
                    self._lines[metric_name] = line
                self.ax.set_title("Dynamical Image of History Training")
                self.ax.set_xlabel("Epochs")
                self.ax.set_ylabel("Loss")
                self.ax.set_yscale("log")
                self.ax.legend(loc="lower left")
            else:
                # Subsequent calls: update existing lines in-place: no artist teardown.
                for loss_data, metric_name in zip(losses, metrics_to_plot):
                    epochs = np.array(loss_data["epochs"])[~np.isnan(loss_data["values"])]
                    values = np.array(loss_data["values"])[~np.isnan(loss_data["values"])]
                    self._lines[metric_name].set_data(epochs, values)
                self.ax.relim()
                self.ax.autoscale_view()

            # Reference lines drawn once via axhline (always spans full x range).
            if self.base_error is not None and "_base_error_line" not in self._lines:
                self._lines["_base_error_line"] = self.ax.axhline(
                    y=self.base_error,
                    color="blue",
                    linestyle=(0, (1, 10)),
                    linewidth=2.5,
                    label="Target Train-Loss Proxy",
                )
                self.ax.legend(loc="lower left")
            if self.base_error_histo is not None and "_base_error_histo_line" not in self._lines:
                self._lines["_base_error_histo_line"] = self.ax.axhline(
                    y=self.base_error_histo,
                    color="green",
                    linestyle=(0, (1, 10)),
                    linewidth=2.5,
                    label="Target Histo Proxy",
                )
                self.ax.legend(loc="lower left")

            plt.pause(0.001)
        if self.output_dir is not None:
            savefig(self.fig, self.output_dir + f"loss_history.svg")
        return
