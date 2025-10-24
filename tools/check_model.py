#!/usr/bin/env python3
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-RemoveBackground_SET
#
# Tool to show PyTorch class i.e:
# python tool/show_class.py -m source/inference/MDX_Net.py:MDX_Net
import argparse
import hashlib
import safetensors.torch
from seconohe.logger import logger_set_standalone
import torch
# Local imports
import bootstrap  # noqa: F401
from src.nodes import main_logger
from src.nodes.utils.misc import cli_add_verbose, cli_add_version
from src.nodes.utils.arch import RemBg


def show_keys_detailed(state_dict):
    """
    Logs detailed information about a PyTorch state_dict, dynamically
    adjusting column width to fit the longest key.
    """
    if not state_dict:
        main_logger.info("--- PyTorch Model State Dict is empty ---")
        return

    main_logger.info("\n--- PyTorch Model State Dict ---")

    # 1. First, find the length of the longest key.
    # We also compare with the header "Key" to ensure the column is at least that wide.
    # Add a couple of spaces for padding.
    max_key_len = max([len(key) for key in state_dict.keys()] + [len("Key")])
    key_col_width = max_key_len + 2

    # Define the other column widths (these can often remain fixed)
    shape_col_width = 25
    num_elem_col_width = 15

    # 2. Build the header using the dynamically calculated width
    header = (
        f"{'Key':<{key_col_width}} | {'Shape':>{shape_col_width}} | "
        f"{'Num Elements':>{num_elem_col_width}} | {'SHA256 (start)'}"
    )
    main_logger.info(header)
    main_logger.info("-" * len(header))

    # 3. Iterate and print the data, using the same dynamic width
    total_el = 0
    for key, tensor in state_dict.items():
        num_elements = tensor.numel()
        total_el += num_elements
        tensor_bytes = tensor.cpu().numpy().tobytes()
        sha256_hash = hashlib.sha256(tensor_bytes).hexdigest()
        short_hash = sha256_hash[:8]

        # Use the calculated key_col_width in the f-string
        log_line = (
            f"{key:<{key_col_width}} | {str(list(tensor.shape)):>{shape_col_width}} | "
            f"{num_elements:>{num_elem_col_width},} | {short_hash}"
        )
        main_logger.info(log_line)

    main_logger.info("-" * len(header))
    main_logger.info(f"{'Total':<{key_col_width}} | {'':>{shape_col_width}} | {total_el:>{num_elem_col_width},} |")


def show_keys(state_dict):
    main_logger.info("\n--- PyTorch Model Keys ---\n\n")
    for key, tensor in state_dict.items():
        main_logger.info(f"{key} {list(tensor.shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check if this is a known BiRefNet model",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('input_file', type=str, help="Path to the input safetensors/PyTorch model file.")
    parser.add_argument('-k', '--keys', action='store_true', help="Print the keys for the state_dict.")
    parser.add_argument('-K', '--detailed-keys', action='store_true', help="Print the keys for the state_dict with detail.")
    cli_add_verbose(parser)
    cli_add_version(parser, __name__)

    args = parser.parse_args()
    logger_set_standalone(main_logger, args)

    model_path = args.input_file
    main_logger.info(f"Analyzing: {model_path}")
    if model_path.endswith(".safetensors"):
        state_dict = safetensors.torch.load_file(model_path)
    else:
        state_dict = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in state_dict:
            # BEN
            state_dict = state_dict['model_state_dict']
            # safetensors.torch.save_file(state_dict, "model.safetensors")
    # Optional print keys
    if args.keys:
        show_keys(state_dict)
    if args.detailed_keys:
        show_keys_detailed(state_dict)

    bb_a = RemBg(state_dict, main_logger, model_path, vae=False)
    bb_a.check()
    main_logger.info(f"Model type: {bb_a.model_type} ({bb_a.dtype})")
    main_logger.info(f"Back bone type: {bb_a.bb}")
    main_logger.info(f"Model version: {bb_a.version}")
