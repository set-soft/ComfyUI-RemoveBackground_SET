#!/usr/bin/env python3
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-BiRefNet-SET
#
# Tool to convert the BEN1/2 to a safetensors file
import argparse
import safetensors.torch
from seconohe.logger import logger_set_standalone
import torch
# Local imports
import bootstrap  # noqa: F401
from src.nodes import main_logger
from src.nodes.utils.misc import cli_add_verbose, cli_add_version
from src.nodes.utils.arch import RemBgArch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BEN1/2 to safetensors",
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
        if 'model_state_dict' in state_dict:
            # BEN
            state_dict = state_dict['model_state_dict']

    bb_a = RemBgArch(state_dict, main_logger, model_path)
    bb_a.check()
    main_logger.info(f"Model type: {bb_a.model_type}")
    main_logger.info(f"Back bone type: {bb_a.bb}")
    main_logger.info(f"Model version: {bb_a.version}")

    # Remove sideout layers, used for supervised learning
    for k in list(state_dict.keys()):
        if k.startswith('sideout'):
            del state_dict[k]

    model = bb_a.instantiate_model()
    model.load_state_dict(state_dict)
    model.eval()
    if args.half:
        model.to(dtype=torch.float16)
        fp = 'F16'
        dt = 'float16'
    else:
        fp = 'F32'
        dt = 'float32'

    name = "BEN2" if model_path.endswith(".safetensors") else "BEN1"
    if model_path.endswith(".safetensors"):
        ori = "https://huggingface.co/PramaLLC/BEN2/resolve/main/model.safetensors"
    else:
        ori = "https://huggingface.co/PramaLLC/BEN/resolve/main/BEN_Base.pth"

    metadata = {
        "desc": f"{name} base background remover",
        "download": f"https://huggingface.co/set-soft/RemBG/resolve/main/MVANet/{name}_Base_{fp}.safetensors",
        "original": ori,
        "file_t": "safetensors",
        "model_t": "MVANet",
        "name": f"{name}_Base_{fp}",
        "size": "1024",
        "normalize": "ImageNet",
        "dtype": dt,
        "project": "https://github.com/set-soft/ComfyUI-RemoveBackground_SET",
        "ben_variant": "True",
        "sideout_pruned": "True",
    }

    safetensors.torch.save_file(model.state_dict(), args.output_file, metadata=metadata)
