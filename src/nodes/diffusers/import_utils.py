# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Import utilities: Utilities related to imports and our lazy inits.
"""

import importlib.util
import sys

# from huggingface_hub.utils import is_jinja_available  # noqa: F401
# from packaging import version
# from packaging.version import Version, parse

import logging


# The package importlib_metadata is in a different place, depending on the python version.
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


# check whether torch_npu is available
_torch_npu_available = importlib.util.find_spec("torch_npu") is not None
if _torch_npu_available:
    try:
        _torch_npu_version = importlib_metadata.version("torch_npu")
        logger.info(f"torch_npu version {_torch_npu_version} available.")
    except ImportError:
        _torch_npu_available = False


_xformers_available = importlib.util.find_spec("xformers") is not None
try:
    _xformers_version = importlib_metadata.version("xformers")
    logger.debug(f"Successfully imported xformers version {_xformers_version}")
except importlib_metadata.PackageNotFoundError:
    _xformers_available = False


def is_torch_npu_available():
    return _torch_npu_available


def is_xformers_available():
    return _xformers_available
