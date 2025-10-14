#!/usr/bin/env python3
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-RemoveBackground_SET
#
# Tool to convert repack DiffDIS including the "positive" embeddings and metadata
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
    parser = argparse.ArgumentParser(description="DiffDIS repacker",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('input_file', type=str, help="Path to the main model.")
    parser.add_argument('embeddings', type=str, help="Path to the encoded empty prompt.")
    parser.add_argument('output_file', type=str, help="Path to the output safetensors model file.")

    parser.add_argument('-H', '--half', action='store_true', help="Convert to FP16")
    cli_add_verbose(parser)
    cli_add_version(parser, __name__)

    args = parser.parse_args()
    logger_set_standalone(main_logger, args)

    model_path = args.input_file
    main_logger.info(f"Analyzing: {model_path}")
    state_dict = safetensors.torch.load_file(model_path, device="cpu")

    model_path = args.embeddings
    main_logger.info(f"Analyzing: {model_path}")
    state_dict_positive = safetensors.torch.load_file(model_path, device="cpu")

    # Make the embeddings look like a ComfyUI conditioner
    positive = [[state_dict_positive['positive']]]
    # Cheat with the VAE, we still need to use the one from SD Turbo as a separated file
    bb_a = RemBgArch(state_dict, main_logger, model_path, positive=positive, vae=False)
    bb_a.check()
    main_logger.info(f"Model type: {bb_a.model_type}")
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

    metadata = {
        "desc": "DiffDIS base background remover + embeddings",
        "download": f"https://huggingface.co/set-soft/RemBG/resolve/main/DiffDIS/DiffDIS_{fp}.safetensors",
        "original": "https://drive.google.com/drive/folders/1NKmUbn9BiV7xYy_1c2khIBAuOQNuYAdR?usp=sharing",
        "file_t": "safetensors",
        "model_t": "DiffDIS",
        "name": f"DiffDIS_{fp}",
        "size": "1024",
        "normalize": "0.5/0.5",
        "dtype": dt,
        "project": "https://github.com/set-soft/ComfyUI-RemoveBackground_SET",
        "positive": "True",
    }

    safetensors.torch.save_file(model.state_dict(), args.output_file, metadata=metadata)
