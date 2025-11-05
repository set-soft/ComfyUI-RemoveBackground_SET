#!/usr/bin/env python3
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-RemoveBackground_SET
#
# Tool to convert the ESNet to a safetensors file
# First use esnet_join.py to join the two halves
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
    parser.add_argument('input_file', type=str, help="Path to the input PyTorch model file.")
    parser.add_argument('output_file', type=str, help="Path to the output safetensors model file.")

    parser.add_argument('-H', '--half', action='store_true', help="Convert to FP16")
    cli_add_verbose(parser)
    cli_add_version(parser, __name__)

    args = parser.parse_args()
    logger_set_standalone(main_logger, args)

    model_path = args.input_file
    main_logger.info(f"Analyzing: {model_path}")

    if model_path.endswith(".safetensors"):
        state_dict = safetensors.torch.load_file(model_path, device='cpu')
    else:
        state_dict = torch.load(model_path, map_location='cpu')

    bb_a = RemBg(state_dict, main_logger, model_path)
    bb_a.check()
    main_logger.info(f"Model type: {bb_a.model_type}")
    main_logger.info(f"Back bone type: {bb_a.bb}")
    main_logger.info(f"Model version: {bb_a.version}")

    if args.half:
        dtype = torch.float16
        fp = 'F16'
        dt = 'float16'
    else:
        dtype = torch.float32
        fp = 'F32'
        dt = 'float32'

    model = bb_a.instantiate_model(state_dict, dtype=dtype)

    name = "ESNet"
    sname = "ResNet50" if bb_a.bb == 'resnet50' else "Swin384"
    metadata = {
        "desc": f"{name} {sname} background remover",
        "download": f"https://huggingface.co/set-soft/RemBG/resolve/main/ESNet/ESNet_{sname}_{fp}.safetensors",
        "original": "https://pan.baidu.com/s/1Yyvqwwe94NhZEIc1IcKY4g?pwd=mvpl",
        "file_t": "safetensors",
        "model_t": "ESNet",
        "name": f"ESNet_{sname}_{fp}",
        "size": "1280",
        "normalize": "ImageNet",
        "dtype": dt,
        "project": "https://github.com/set-soft/ComfyUI-RemoveBackground_SET",
        "datasets": "HRSOD, DAVIS-SOD, UHRSD",
    }

    safetensors.torch.save_file(model.state_dict(), args.output_file, metadata=metadata)
