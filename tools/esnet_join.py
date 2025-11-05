#!/usr/bin/env python3
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-RemoveBackground_SET
#
# Tool to join the 2 ESNet halves
import argparse
import safetensors.torch
from seconohe.logger import logger_set_standalone
import torch
# Local imports
import bootstrap  # noqa: F401
from src.nodes import main_logger
from src.nodes.utils.misc import cli_add_verbose, cli_add_version
from src.nodes.utils.arch import RemBg


def load_model(fname):
    model_path = fname
    main_logger.info(f"Analyzing: {model_path}")

    if model_path.endswith(".safetensors"):
        state_dict = safetensors.torch.load_file(model_path, device='cpu')
    else:
        state_dict = torch.load(model_path, map_location='cpu')
        if 'net' in state_dict:
            state_dict = state_dict['net']

    bb_a = RemBg(state_dict, main_logger, model_path)
    bb_a.check()
    main_logger.info(f"Model type: {bb_a.model_type}")
    main_logger.info(f"Back bone type: {bb_a.bb}")
    main_logger.info(f"Model version: {bb_a.version}")
    return bb_a, state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESNet to safetensors",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('first_file', type=str, help="Path to the first PyTorch/safetensors model file.")
    parser.add_argument('second_file', type=str, help="Path to the first PyTorch/safetensors model file.")
    parser.add_argument('output_file', type=str, help="Path to the output safetensors model file.")

    parser.add_argument('-H', '--half', action='store_true', help="Convert to FP16")
    cli_add_verbose(parser)
    cli_add_version(parser, __name__)

    args = parser.parse_args()
    logger_set_standalone(main_logger, args)

    first, sd1 = load_model(args.first_file)
    assert first.model_type == 'ESNet_first'
    second, sd2 = load_model(args.second_file)
    assert second.model_type == 'ESNet_second'

    metadata = {}
    sd_first = {'first.'+k: v for k, v in sd1.items()}
    sd_second = {'second.'+k: v for k, v in sd2.items()}
    sd_first.update(sd_second)
    safetensors.torch.save_file(sd_first, args.output_file, metadata=metadata)
