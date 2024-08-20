from typing import Union, Tuple

import torch

from potentials.utils import sum_except_batch, get_batch_shape


def exponential_transform(x: torch.Tensor,
                          batch_shape: Union[Tuple[int, ...], torch.Size]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transform all elements of the input with an exponential function.

    :param x: input parameter tensor with shape `(*batch_shape, *parameter_shape)`.
    :param batch_shape: input tensor batch shape.
    :return: transformed tensor and log determinant of the Jacobian of the transformation.
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    y = torch.exp(x)
    log_det = sum_except_batch(y, batch_shape)
    return y, log_det

