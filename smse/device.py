import torch


def get_device() -> torch.device:
    """
    Get the current device (CPU or GPU) for PyTorch.

    Returns:
        torch.device: The current device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
