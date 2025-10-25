#!/usr/bin/env python3
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-RemoveBackground_SET
#
# Tool to convert the U-2-Net to a safetensors file
import os
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
    parser = argparse.ArgumentParser(description="U²-Net to safetensors",
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

    bb_a = RemBg(state_dict, main_logger, model_path)
    bb_a.check()
    main_logger.info(f"Model type: {bb_a.model_type}")

    if args.half:
        dtype = torch.float16
        fp = 'F16'
        dt = 'float16'
    else:
        dtype = torch.float32
        fp = 'F32'
        dt = 'float32'

    model = bb_a.instantiate_model(state_dict, dtype=dtype)

    basename = os.path.splitext(os.path.basename(model_path))[0]
    name = "U2Net"
    if basename == 'u2net_human_seg':
        ori = "https://drive.google.com/file/d/1m_Kgs91b21gayc2XLW0ou8yugAIadWVP/view?usp=sharing"
        subname = "Human"
    elif basename == 'u2net':
        ori = "https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view?usp=sharing"
        subname = "Base"
        assert not args.half, "Nope, you get NaN"
    elif basename == 'u2netp':
        ori = "https://drive.google.com/file/d/1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy/view?usp=sharing"
        subname = "Small"
        assert not args.half, "Nope, 2 MiB model doesn't work ;-)"
    else:
        assert False, "Add it to the code"

    metadata = {
        "desc": f"{name} {subname.lower()} background remover",
        "download": f"https://huggingface.co/set-soft/RemBG/resolve/main/U2Net/{name}_{subname}_{fp}.safetensors",
        "original": ori,
        "file_t": "safetensors",
        "model_t": "MVANet",
        "name": f"{name}_{subname}_{fp}",
        "size": "320",
        "normalize": "ImageNet",
        "dtype": dt,
        "project": "https://github.com/set-soft/ComfyUI-RemoveBackground_SET",
    }

    safetensors.torch.save_file(model.state_dict(), args.output_file, metadata=metadata)
