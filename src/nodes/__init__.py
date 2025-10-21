# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPL-3.0
# Project: ComfyUI-RemoveBackground_SET
from seconohe.logger import initialize_logger


__version__ = "1.0.0"
__copyright__ = "Copyright © 2025 Salvador E. Tropea / Instituto Nacional de Tecnología Industrial"
__license__ = "License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>"
__author__ = "Salvador E. Tropea"
NODES_NAME = "RemoveBackground_SET"
MODELS_DIR_KEY = "rembg"
MODELS_DIR = "RemBG"
main_logger = initialize_logger(NODES_NAME)

BATCHED_OPS = ("INT", {
                  "default": 1, "min": 1, "max": 256, "step": 1,
                  "tooltip": ("How many images to process at once")})
DEFAULT_UPSCALE = "bicubic"
CATEGORY_BASE = "RemBG_SET"
CATEGORY_BASIC = CATEGORY_BASE+"/Basic"
CATEGORY_LOAD = CATEGORY_BASE+"/Load"
CATEGORY_ADV = CATEGORY_BASE+"/Advanced"
