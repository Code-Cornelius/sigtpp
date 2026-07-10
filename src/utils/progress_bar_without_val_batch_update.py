import sys

from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm, TQDMProgressBar
from tqdm import tqdm


class ProgressbarWithoutValBatchUpdate(TQDMProgressBar):
    """
    This progressbar has disabled the validation tqdm bar, which is not useful when there is not a lot of data.
    We have also removed the `v_num` value that appears in the metric and which is constant through the training.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_metrics(self, *args, **kwargs):
        # References: https://lightning.ai/docs/pytorch/stable/extensions/logging.html

        # don't show the version number
        items = super().get_metrics(*args, **kwargs)
        items.pop("v_num", None)
        return items

    def init_validation_tqdm(self):
        """Override this to customize the tqdm bar for validation."""
        bar = tqdm(disable=True)
        return bar

    def init_train_tqdm(self):
        """Override this to customize the tqdm bar for training."""
        bar = Tqdm(
            desc="Training",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            colour='blue',
        )
        return bar
