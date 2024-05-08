from typing import Tuple

import torch


def linear_get_scale_and_zero_point(tensor: torch.tensor, dtype: torch.dtype) -> Tuple[float, int]:
    """Get the scale and zero point for a linear quantization.

    Parameters
    ----------
    tensor: torch.tensor
        The tensor to quantize.
    dtype: torch.dtype
        The dtype to quantize to.
    Returns
    -------
    Tuple[float, int]
        Scale and zero point for the quantization.
    """
    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max

    r_min = tensor.min().item()
    r_max = tensor.max().item()

    scale = (r_max - r_min) / (q_max - q_min)
    zero_point = q_min - r_min / scale

    if zero_point < q_min:
        zero_point = q_min
    elif zero_point > q_max:
        zero_point = q_max
    else:
        zero_point = int(round(zero_point))

    return scale, zero_point


def symmetric_get_scale(tensor: torch.tensor, dtype: torch.dtype) -> float:
    """Get the scale for a symmetric quantization.

    Parameters
    ----------
    tensor: torch.tensor
        The tensor to quantize.
    dtype: torch.dtype
        The dtype to quantize to.
    Returns
    -------
    float
        Scale for the quantization.
    """
    q_max = torch.iinfo(dtype).max
    r_max = tensor.abs().max().item()

    scale = r_max / q_max
    return scale


def linear_q_scale_zero_point(tensor: torch.tensor, scale: float, zero_point: int, dtype: torch.dtype) -> torch.tensor:
    """Quantize a tensor to a given scale and zero point.

    Parameters
    ----------
    tensor: torch.tensor
        The tensor to quantize
    scale: float
        The scale
    zero_point: int
        The zero point
    dtype: torch.dtype
        The dtype to quantize to
    Returns
    -------
    torch.tensor
        The quantized tensor
    """
    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max

    q = tensor / scale + zero_point
    q = torch.round(q)
    q = q.clamp(q_min, q_max).to(dtype)

    return q
