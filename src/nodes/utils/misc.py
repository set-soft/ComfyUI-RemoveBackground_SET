# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-RemoveBackground_SET
import argparse
from .. import __version__, __copyright__, __license__, __author__, NODES_NAME


def cli_add_verbose(parser):
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help="Enable verbose output to see details of the process.")


class PrintVersionAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        # Format the version information
        version_info = f"""{parser.prog} ({NODES_NAME}) {__version__}
{__copyright__}
{__license__}
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.

Written by {__author__}"""
        print(version_info)
        # Exit the parser
        parser.exit()


def cli_add_version(parser, prog_name):
    parser.add_argument('-V', '--version', help="Show version and copyright information and exit",
                        action=PrintVersionAction)
