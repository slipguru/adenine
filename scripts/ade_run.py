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

from adenine.core import define_pipeline
from adenine.core import pipelines
from adenine.utils import extra


def main(config_file):
    """Generate the pipelines."""
    # Load the configuration file
    config_path = os.path.abspath(config_file)
    config = imp.load_source('ade_config', config_path)
    extra.set_module_defaults(config, {'step0': {'Impute': [False]},
                                       'step1': {'None': [True]},
                                       'step2': {'None': [True]},
                                       'step3': {'None': [False]},
                                       'exp_tag': 'debug',
                                       'output_root_folder': 'results'})

    # Read the variables from the config file
    X, y = config.X, config.y

    # Get the experiment tag and the output root folder
    exp_tag, root = config.exp_tag, config.output_root_folder
    if not os.path.exists(root):
        os.makedirs(root)

    # Define the ade.log file (a new one for each run)
    filename = '_'.join(('ade', exp_tag, extra.get_time()))
    logfile = os.path.join(root, filename + '.log')
    logging.basicConfig(filename=logfile, level=logging.INFO, filemode='w',
                        format='%(levelname)s (%(name)s): %(message)s')
    root_logger = logging.getLogger()
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    ch.setFormatter(logging.Formatter('%(levelname)s (%(name)s): %(message)s'))
    root_logger.addHandler(ch)

    # Pipelines Definition
    pipes = define_pipeline.parse_steps([config.step0, config.step1,
                                         config.step2, config.step3])

    # Pipelines Evaluation
    out_folder = pipelines.run(pipes=pipes, X=X, exp_tag=filename,
                               root=root, y=y)

    # Copy the ade_config just used into the outFolder
    shutil.copy(config_path, os.path.join(out_folder, 'ade_config.py'))
    # Move the logging file into the outFolder
    shutil.move(logfile, out_folder)

# ----------------------------  RUN MAIN ---------------------------- #
if __name__ == '__main__':
    from adenine import __version__
    parser = argparse.ArgumentParser(description='Adenine script for '
                                                 'pipeline generation.')
    parser.add_argument('--version', action='version',
                        version='%(prog)s v'+__version__)
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
