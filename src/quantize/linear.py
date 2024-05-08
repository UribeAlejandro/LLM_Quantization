from typing import Tuple

import torch

from src.quantize.utils import linear_get_scale_and_zero_point, linear_q_scale_zero_point, symmetric_get_scale


def linear_quantization(tensor: torch.Tensor, dtype: torch.dtype = torch.int8) -> Tuple[torch.Tensor, float, int]:
    """Quantize a tensor to a given dtype.

    Parameters
    ----------
    tensor: torch.tensor
        The tensor to quantize
    dtype: torch.dtype (default: torch.int8)
        The dtype to quantize to
    Returns
    -------
    Tuple[torch.tensor, float, int]
        The quantized tensor, the scale and the zero point
    """
    scale, zero_point = linear_get_scale_and_zero_point(tensor, dtype)
    q = linear_q_scale_zero_point(tensor, scale, zero_point, dtype)

    return q, scale, zero_point


def linear_symmetric_quantization(tensor: torch.Tensor, dtype: torch.dtype = torch.int8) -> Tuple[torch.Tensor, float]:
    """Quantize a tensor to a given dtype.

    Parameters
    ----------
    tensor: torch.tensor
        The tensor to quantize
    dtype: torch.dtype (default: torch.int8)
        The dtype to quantize to
    Returns
    -------
    Tuple[torch.tensor, float]
        The quantized tensor and the scale
    """
    zero_point = 0
    scale = symmetric_get_scale(tensor, dtype)
    q = linear_q_scale_zero_point(tensor, scale, zero_point, dtype)

    return q, scale


def dequantization(tensor: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    """Dequantize a tensor.

    Parameters
    ----------
    tensor: torch.tensor
        The tensor to dequantize
    scale: float
        The scale
    zero_point: int
        The zero point

    Returns
    -------
    torch.tensor
        The dequantized tensor
    """
    return scale * (tensor.float() - zero_point)
