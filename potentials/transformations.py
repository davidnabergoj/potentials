import math
from typing import Union, Tuple

import torch

from potentials.utils import sum_except_batch


def exponential_transform(x: torch.Tensor,
                          batch_shape: Union[Tuple[int, ...], torch.Size]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transform all elements of the input with an exponential function.
    The output range is `(0, inf)`.

    :param torch.Tensor x: input parameter tensor with shape `(*batch_shape, *parameter_shape)`.
    :param Union[Tuple[int, ...], torch.Size] batch_shape: input tensor batch shape.
    :return: transformed tensor and log determinant of the Jacobian of the transformation.
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    y = torch.exp(x)
    log_det = sum_except_batch(y, batch_shape)
    return y, log_det


def affine_transform(x: torch.Tensor,
                     batch_shape: Union[Tuple[int, ...], torch.Size],
                     scale: float,
                     shift: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transform all elements of the input with an affine transform.
    The output range is `(-inf, inf)`.

    :param torch.Tensor x: input parameter tensor with shape `(*batch_shape, *parameter_shape)`.
    :param Union[Tuple[int, ...], torch.Size] batch_shape: input tensor batch shape.
    :param float scale: scale value.
    :param float shift: shift value.
    :return: transformed tensor and log determinant of the Jacobian of the transformation.
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    y = x * scale + shift
    log_det = sum_except_batch(
        torch.full(size=x.shape, fill_value=math.log(scale), dtype=x.dtype, device=x.device),
        batch_shape
    )
    return y, log_det


def sigmoid_transform(x: torch.Tensor,
                      batch_shape: Union[Tuple[int, ...], torch.Size]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transform all elements of the input with a sigmoid transform.
    The output range is `(0, 1)`.

    :param torch.Tensor x: input parameter tensor with shape `(*batch_shape, *parameter_shape)`.
    :param Union[Tuple[int, ...], torch.Size] batch_shape: input tensor batch shape.
    :return: transformed tensor and log determinant of the Jacobian of the transformation.
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    y = torch.sigmoid(x)
    log_det = sum_except_batch(
        y * (1 - y),
        batch_shape
    )
    return y, log_det


def negative_exponential_transform(x: torch.Tensor,
                                   batch_shape: Union[Tuple[int, ...], torch.Size]) -> Tuple[
    torch.Tensor, torch.Tensor]:
    """
    Transform all elements of the input `x` to `-exp(x)`.
    The output range is `(-inf, 0)`.

    :param torch.Tensor x: input parameter tensor with shape `(*batch_shape, *parameter_shape)`.
    :param Union[Tuple[int, ...], torch.Size] batch_shape: input tensor batch shape.
    :return: transformed tensor and log determinant of the Jacobian of the transformation.
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    z, log_det_0 = exponential_transform(x, batch_shape)
    y, log_det_1 = affine_transform(z, batch_shape, scale=-1.0, shift=0.0)
    log_det = log_det_0 + log_det_1
    return y, log_det


def scaled_sigmoid_transform(x: torch.Tensor,
                             batch_shape: Union[Tuple[int, ...], torch.Size],
                             low: float,
                             high: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transform all elements of the input tensor with a scaled sigmoid.
    The output range is `(-inf, 0)`.

    :param torch.Tensor x: input parameter tensor with shape `(*batch_shape, *parameter_shape)`.
    :param Union[Tuple[int, ...], torch.Size] batch_shape: input tensor batch shape.
    :param float low: lower bound value.
    :param float high: upper bound value.
    :return: transformed tensor and log determinant of the Jacobian of the transformation.
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    z, log_det_0 = sigmoid_transform(x, batch_shape)
    y, log_det_1 = affine_transform(z, batch_shape, scale=(high - low), shift=low)
    log_det = log_det_0 + log_det_1
    return y, log_det


def bound_parameter(x: torch.Tensor,
                    batch_shape: Union[Tuple[int, ...], torch.Size],
                    low: Union[float],
                    high: Union[float]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Bound an input parameter to a desired range.

    :param torch.Tensor x: input parameter tensor with shape `(*batch_shape, *parameter_shape)`.
    :param Union[Tuple[int, ...], torch.Size] batch_shape: input tensor batch shape.
    :param float low: lower bound value.
    :param float high: upper bound value.
    :return: transformed tensor and log determinant of the Jacobian of the transformation.
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    if low >= high:
        raise ValueError(f'Lower bound {low} is greater than or equal to high {high}')
    if low == torch.inf:
        raise ValueError(f'Lower bound {low} cannot be infinity')
    if high == -torch.inf:
        raise ValueError(f'Higher bound {high} cannot be negative infinity')

    if low == -torch.inf and high < torch.inf:
        return negative_exponential_transform(x, batch_shape)
    elif low > -torch.inf and high == torch.inf:
        return exponential_transform(x, batch_shape)
    elif low > -torch.inf and high < torch.inf:
        return scaled_sigmoid_transform(x, batch_shape, low, high)
    else:
        raise ValueError
