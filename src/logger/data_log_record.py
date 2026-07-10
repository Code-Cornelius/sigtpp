import logging

import numpy as np
import numpy.typing
import torch


class DataLogRecord(logging.LogRecord):
    # Bug found: when using f-strings and there is number in one of the strings passed, it makes an error.

    NUM_ELEMENTS_SHOWN = 2

    # The advantage of this logRecord is that we lazily transform the data for logging.
    # This is particularly useful when logs are called in computationally expensive areas of the code.
    # To use it, log the data with the following syntax:
    # logger.info("tensor %s", tensor)
    # do not use f-strings.
    def getMessage(self):
        msg = str(self.msg)
        with np.printoptions(precision=4, suppress=True, linewidth=115):
            if self.args:
                processed_args = []
                for arg in self.args:
                    if isinstance(arg, torch.Tensor):
                        arg = arg.detach().cpu().numpy()
                    if isinstance(arg, np.ndarray):
                        processed_args.append(self._format_array(arg))
                    else:
                        processed_args.append(arg)
                msg = msg % tuple(processed_args)
        return msg

    def _format_array(self, array: np.typing.ArrayLike):
        shape = list(array.shape)
        total_elements = np.prod(shape)
        nan_count = np.isnan(array).sum()

        if len(shape) == 3:
            # Check for NaNs
            if nan_count > 0:
                means = np.nanmean(array, axis=(0, 1))
                stds = np.nanstd(array, axis=(0, 1))
                nan_info = f" ({(nan_count / total_elements) * 100:.2f}% of values are NaNs - {nan_count} elements)."
            else:
                means = np.mean(array, axis=(0, 1))
                stds = np.std(array, axis=(0, 1))
                nan_info = "."

            # In the case when the sequence is very long, we only show the first element because otherwise it will clog the logs.
            first_batch_elements = (
                np.array2string(array[: DataLogRecord.NUM_ELEMENTS_SHOWN, :, :].transpose(0, 2, 1))
                if shape[1] < 120
                else np.array2string(array[:1, :300, :].transpose(0, 2, 1))
            )
            return (
                f"- shape: {shape} with means (last dimension): "
                + np.array2string(means)
                + " and stds: "
                + np.array2string(stds)
                + f"{nan_info} {'First batch element' if shape[1] > 120 else str(DataLogRecord.NUM_ELEMENTS_SHOWN) + ' first batch elements'}: \n"
                + first_batch_elements
            )
        else:
            return np.array2string(array)


if __name__ == "__main__":
    import logging.config

    from src.logger.init_logger import set_config_logging

    set_config_logging()
    logger = logging.getLogger(__name__)

    tensor = torch.rand(2, 10, 3)
    logger.info("tensor %s", tensor)
    logger.info("array %s", tensor.numpy())
