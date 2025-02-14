from typing import Dict, List, Tuple, Type, Union
import torch as th
import torch.nn as nn

class SmoothReLU(nn.Module):
    def __init__(self, d=0.2):
        """
        Custom SmoothReLU class that satisfies the following piecewise conditions:
        
        f(x) = 0, x <= 0
        f(x) = x^2 / (2 * d), 0 < x < d
        f(x) = x - d / 2, x >= d
        Args:
            d (float): Controls the smoothness and the transition point.
        """
        super(SmoothReLU, self).__init__()
        self.d = d

    def forward(self, x):
        # Define the piecewise function
        result = th.where(
            x <= 0, 
            th.zeros_like(x),  # Value when x <= 0
            th.where(
                x < self.d,
                x**2 / (2 * self.d),  # Value when 0 < x < d
                x - self.d / 2  # Value when x >= d
            )
        )
        return result

def get_device(device: Union[th.device, str] = "auto") -> th.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    :param device: One for 'auto', 'cuda', 'cpu'
    :return: Supported Pytorch device
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to th.device
    device = th.device(device)

    # Cuda not available
    if device.type == th.device("cuda").type and not th.cuda.is_available():
        return th.device("cpu")

    return device