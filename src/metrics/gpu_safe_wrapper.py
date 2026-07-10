import logging

import torch

logger = logging.getLogger(__name__)


def gpu_memory_safe(func):
    """
    Wraps a function to safely catch CUDA out-of-memory errors.
    Returns torch.nan if OOM occurs, otherwise returns the function's output.
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()  # optional, frees memory for next calls
                logger.error(f"CUDA out of memory error caught in {func.__name__}: {e}")
                return torch.tensor(float('nan'), device='cpu')
            else:
                raise  # re-raise unexpected RuntimeErrors

    return wrapper


if __name__ == "__main__":

    @gpu_memory_safe
    def my_heavy_gpu_op(x):
        return torch.mm(x, x)  # example operation

    # Test:
    x = torch.randn(50000, 50000, device='cuda')  # likely to OOM
    result = my_heavy_gpu_op(x)
    print(result)  # Should print 'nan' if OOM occurs
