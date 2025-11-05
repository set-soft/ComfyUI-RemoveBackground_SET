# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-RemoveBackground_SET
import argparse
import torch
from .. import __version__, __copyright__, __license__, __author__, NODES_NAME


def cli_add_verbose(parser):
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help="Enable verbose output to see details of the process.")


class PrintVersionAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        # Format the version information
        version_info = f"""{parser.prog} ({NODES_NAME}) {__version__}
{__copyright__}
{__license__}
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.

Written by {__author__}"""
        print(version_info)
        # Exit the parser
        parser.exit()


def cli_add_version(parser, prog_name):
    parser.add_argument('-V', '--version', help="Show version and copyright information and exit",
                        action=PrintVersionAction)


def batched_min_max_norm(s: torch.Tensor, in_place: bool = True) -> torch.Tensor:
    """
    Performs per-sample min-max normalization on a batch of tensors.

    Each sample in the batch (along the 0-th dimension) is independently
    normalized to a [0, 1] range. This function is fully vectorized for
    maximum performance.

    Args:
        s (torch.Tensor): The input tensor, expected to have a shape of
                          (B, C, H, W) or similar batch format.
        in_place (bool, optional): If True, the input tensor `s` will be
                                   modified directly to save memory.
                                   If False, a clone of `s` is created and
                                   returned. Defaults to True.

    Returns:
        torch.Tensor: The normalized tensor. This will be the same object
                      as the input `s` if `in_place=True`.
    """
    if not in_place:
        s = s.clone()

    B = s.shape[0]

    # Flatten the spatial/channel dimensions to find min/max for each sample
    s_flat = s.view(B, -1)
    mi = torch.min(s_flat, dim=1, keepdim=True).values
    ma = torch.max(s_flat, dim=1, keepdim=True).values

    # Reshape min/max for broadcasting
    mi = mi.view(B, 1, 1, 1)
    ma = ma.view(B, 1, 1, 1)

    # Calculate the denominator, handling the case where max == min
    denominator = ma - mi + 1e-8

    # Apply the normalization in-place
    s -= mi
    s /= denominator

    return s


def sigmoid_and_batched_min_max_norm(s: torch.Tensor, in_place: bool = True) -> torch.Tensor:
    """
    Applies a sigmoid function and then per-sample min-max normalization.

    This function combines sigmoid and normalization into a single utility,
    correctly handling in-place and out-of-place operations.

    Args:
        s (torch.Tensor): The input tensor.
        in_place (bool, optional): If True, the input tensor `s` will be
                                   modified directly. If False, a new tensor
                                   is created. Defaults to True.

    Returns:
        torch.Tensor: The transformed tensor.
    """
    if in_place:
        # Apply sigmoid directly to the input tensor
        s.sigmoid_()
    else:
        # Create a new tensor for the sigmoid result
        s = torch.sigmoid(s)

    # Normalize the result. It's always safe to do this part in-place,
    # because if in_place=False, 's' is already a new tensor.
    return batched_min_max_norm(s, True)
