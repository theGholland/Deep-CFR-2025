import torch


def resolve_device(dev):
    """Resolve a device string into a :class:`torch.device`.

    Args:
        dev (str or torch.device): Desired device specification such as
            "cpu", "cuda", or "auto". If "auto" is given, the function
            selects "cuda" when a GPU is available, otherwise "cpu".

    Returns:
        torch.device: The resolved device.
    """
    if isinstance(dev, torch.device):
        return dev
    if isinstance(dev, str):
        dev = dev.lower()
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(dev)
    raise TypeError("Device must be a string or torch.device")
