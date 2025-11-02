#!/usr/bin/env python3
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-RemoveBackground_SET
#
# Tool to convert the RMFormer to a safetensors file
import argparse
import safetensors.torch
from seconohe.logger import logger_set_standalone
import torch
# Local imports
import bootstrap  # noqa: F401
from src.nodes import main_logger
from src.nodes.utils.misc import cli_add_verbose, cli_add_version
from src.nodes.utils.arch import RemBg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RMFormer to safetensors",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('input_file', type=str, help="Path to the input PyTorch/safetensors model file.")
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
        if 'net' in state_dict:
            state_dict = state_dict['net']

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

    name = "RMFormer"
    if '_KUH_' in model_path:
        datasets = 'HRS10K, UHRSD, HRSOD'
        sname = 'KUH'
    elif '_UH_' in model_path:
        datasets = 'UHRSD, HRSOD'
        sname = 'UH'
    elif '_DH_' in model_path:
        datasets = 'DUTS, HRSOD'
        sname = 'DH'
    metadata = {
        "desc": f"{name} {datasets} background remover",
        "download": f"https://huggingface.co/set-soft/RemBG/resolve/main/PGNet/RMFormer_{sname}_{fp}.safetensors",
        "original": "https://drive.google.com/drive/folders/17LkT_7GHMiQ2Eqnj3aBTqUyUdVszd9x2?usp=drive_link",
        "file_t": "safetensors",
        "model_t": "RMFormer",
        "name": f"RMFormer_{sname}_{fp}",
        "size": "1536",
        "normalize": "ImageNet",
        "dtype": dt,
        "project": "https://github.com/set-soft/ComfyUI-RemoveBackground_SET",
        "epoch": "final",
    }

    model.eval()
    # This model uses shared buffers, so we avoid saving them repeated times
    # https://huggingface.co/docs/safetensors/torch_shared_tensors
    safetensors.torch.save_model(model, args.output_file, metadata=metadata)
