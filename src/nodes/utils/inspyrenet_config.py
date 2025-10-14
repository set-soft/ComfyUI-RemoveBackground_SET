# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-RemoveBackground_SET
#
# Simple parser to get the transparent-background (InSPyReNet) definitions
# We don't need transparent-background, but if installed we can recycle the downloaded models
# The code was created using Gemini 2.5 Pro assistance.
# YAML parser is optional.
#
from dataclasses import dataclass, field
import logging
import os
import re
from typing import List, Optional

try:
    import yaml
    HAVE_PYYAML = True
except ImportError:
    HAVE_PYYAML = False


@dataclass
class InSPyReNetModelCfg:
    """
    A dataclass to hold the configuration for a single InSPyReNet model checkpoint.
    """
    # Required field
    name: str
    # Optional fields that will be filled in during parsing
    url: Optional[str] = None
    ckpt_name: Optional[str] = None
    base_size: List[int] = field(default_factory=list)


# Regex to capture the model name (e.g., "fast:")
TB_MODEL_NAME = re.compile(r'^(\S+):\s*$')
# Regex to capture the URL string
TB_URL = re.compile(r'^\s+url:\s*"?([^"]+)"?\s*$')
# Regex to capture the ckpt_name string
TB_CKPT_NAME = re.compile(r'^\s+ckpt_name:\s*"?([^"]+)"?\s*$')
# Regex to capture the base_size list of two integers
TB_BASE_SIZE = re.compile(r'^\s+base_size:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]\s*$')


def _parse_with_pyyaml(file_path: str, home_dir: str) -> List[InSPyReNetModelCfg]:
    """ Parses the config file using the robust PyYAML library. """
    models: List[InSPyReNetModelCfg] = []
    with open(file_path, 'r') as f:
        raw_configs = yaml.safe_load(f)
        if not raw_configs:
            return []

        for name, data in raw_configs.items():
            # Manually map fields to avoid errors if the YAML has extra keys
            # .get() is used for safety in case a field is missing
            ckpt_name = data.get('ckpt_name')
            url = data.get('url')
            if not ckpt_name or not url:
                continue
            models.append(InSPyReNetModelCfg(name=name,
                                             url=url,
                                             ckpt_name=os.path.abspath(os.path.join(home_dir, ckpt_name)),
                                             base_size=data.get('base_size', [1024, 1024])))
    return models


def _parse_with_regex(file_path: str, home_dir: str) -> List[InSPyReNetModelCfg]:
    """ Parses the config file using line-by-line regex matching as a fallback. """
    models: List[InSPyReNetModelCfg] = []
    cur_model: Optional[InSPyReNetModelCfg] = None

    with open(file_path) as f:
        for line in f:
            res = TB_MODEL_NAME.match(line)
            if res:
                # Found a new model definition.
                # First, save the previously parsed model if it exists.
                if cur_model and cur_model.url and cur_model.ckpt_name:
                    models.append(cur_model)

                # Start a new, empty model config
                cur_model = InSPyReNetModelCfg(name=res.group(1))
                continue

            # Skip any lines that come before the first model name
            if not cur_model:
                continue

            # Try to match the indented property lines
            res = TB_URL.match(line)
            if res:
                cur_model.url = res.group(1)
                continue

            res = TB_CKPT_NAME.match(line)
            if res:
                cur_model.ckpt_name = os.path.abspath(os.path.join(home_dir, res.group(1)))
                continue

            res = TB_BASE_SIZE.match(line)
            if res:
                # Capture the two numbers, convert them to int, and store as a list
                cur_model.base_size = [int(res.group(1)), int(res.group(2))]
                continue

        # After the loop finishes, add the last parsed model
        if cur_model and cur_model.url and cur_model.ckpt_name:
            models.append(cur_model)
    return models


def parse_inspyrenet_config(logger: logging.Logger) -> List[InSPyReNetModelCfg]:
    """
    Parses the transparent-background config.yaml file to extract model information.
    Uses the PyYAML library if installed for robust parsing, otherwise falls back
    to a line-by-line regex-based parser.

    Returns:
        A list of InSPyReNetModelCfg objects found in the config file.
    """
    # Determine the config file path, defaulting to the user's home directory
    cfg_path = os.environ.get('TRANSPARENT_BACKGROUND_FILE_PATH', os.path.abspath(os.path.expanduser('~')))
    home_dir = os.path.join(cfg_path, ".transparent-background")
    cfg_file = os.path.join(home_dir, "config.yaml")

    if not os.path.isfile(cfg_file):
        logger.debug(f"Could not find transparent-background config file at '{cfg_file}'.")
        return []

    logger.debug(f"Looking for InSPyReNet models defined in '{cfg_file}'.")
    try:
        if HAVE_PYYAML:
            logger.debug("Using PyYAML parser.")
            models = _parse_with_pyyaml(cfg_file, home_dir)
        else:
            logger.debug("PyYAML not found. Using fallback regex parser."
                         " For more robust parsing, please `pip install pyyaml`.")
            models = _parse_with_regex(cfg_file, home_dir)
    except Exception as e:
        logger.warning(f"Error parsing config file '{cfg_file}': {e}")
        return []
    logger.debug(f"Found {len(models)} models.")

    return models
