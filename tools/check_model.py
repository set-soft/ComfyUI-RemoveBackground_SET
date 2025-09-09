#!/usr/bin/env python3
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-BiRefNet-SET
#
# Tool to show PyTorch class i.e:
# python tool/show_class.py -m source/inference/MDX_Net.py:MDX_Net
import argparse
import safetensors.torch
from seconohe.logger import logger_set_standalone
import torch
# Local imports
import bootstrap  # noqa: F401
from src.nodes import main_logger
from src.nodes.utils.misc import cli_add_verbose, cli_add_version
from src.nodes.utils.arch import BiRefNetArch


def show_keys(state_dict):
    main_logger.info("\n--- PyTorch Model Keys ---\n\n")
    for key in state_dict.keys():
        main_logger.info(key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check if this is a known BiRefNet model",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('input_file', type=str, help="Path to the input safetensors/PyTorch model file.")
    parser.add_argument('-k', '--keys', action='store_true', help="Print the keys for the state_dict.")
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
    # Optional print keys
    if args.keys:
        show_keys(state_dict)

    bb_a = BiRefNetArch(state_dict, main_logger)
    bb_a.check()
    main_logger.info(f"Back bone type: {bb_a.bb}")
    main_logger.info(f"Model version: {bb_a.version}")
