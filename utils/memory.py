import psutil


def get_available_ram() -> int:
    """Return available RAM in bytes."""
    return psutil.virtual_memory().available


def estimate_batch_size(min_batch_size: int = 512, max_batch_size: int = 16384) -> int:
    """Estimate a batch size from available RAM.

    The estimate scales linearly with available gigabytes (1GB -> 1024 samples)
    and is clamped between ``min_batch_size`` and ``max_batch_size``.
    """
    avail_gb = get_available_ram() / (1024 ** 3)
    est = int(avail_gb * 1024)
    if max_batch_size is not None:
        est = min(est, max_batch_size)
    return max(min_batch_size, est)
