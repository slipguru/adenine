#!/usr/bin/env python
# -*- coding: utf-8 -*-

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Federico Tomasi, Annalisa Barla
#
# FreeBSD License
######################################################################

import imp
import os
import shutil
import logging
import argparse

from adenine import main
from adenine.core import define_pipeline
from adenine.core import pipelines
from adenine.utils import extra


def init_main():
    from adenine import __version__
    parser = argparse.ArgumentParser(description='Adenine script for '
                                                 'pipeline generation.')
    parser.add_argument('--version', action='version',
                        version='%(prog)s v' + __version__)
    parser.add_argument("-c", "--create", dest="create", action="store_true",
                        help="create config file", default=False)
    parser.add_argument("configuration_file", help="specify config file",
                        default='ade_config.py')
    args = parser.parse_args()

    if args.create:
        import adenine as ade
        std_config_path = os.path.join(ade.__path__[0], 'ade_config.py')
        # Check for .pyc
        if std_config_path.endswith('.pyc'):
            std_config_path = std_config_path[:-1]
        # Check if the file already exists
        if os.path.exists(args.configuration_file):
            parser.error("adenine configuration file already exists")
        # Copy the config file
        shutil.copy(std_config_path, args.configuration_file)
    else:
        main(args.configuration_file)


if __name__ == '__main__':
    init_main()
