#!/usr/bin/env python
# -*- coding: utf-8 -*-

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Federico Tomasi, Annalisa Barla
#
# FreeBSD License
######################################################################

import imp, sys, os, shutil
import time
import logging
import argparse

from adenine.core import define_pipeline
from adenine.core import pipelines
from adenine.utils import extra

def main(config_file):
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
    X, y, feat_names, class_names = config.X, config.y, config.feat_names, config.class_names

    # Get the experiment tag and the output root folder
    exp_tag, root = config.exp_tag, config.output_root_folder
    if not os.path.exists(root):
        os.makedirs(root)

    # Define the ade.log file (a new one for each run)
    fileName = '_'.join(('ade', exp_tag, extra.get_time()))
    logFileName = os.path.join(root, fileName+'.log')
    logging.basicConfig(filename=logFileName, level=logging.INFO, filemode='w',
                        format='%(levelname)s (%(name)s): %(message)s')
    root_logger = logging.getLogger()
    ch = logging.StreamHandler()
    ch.setLevel(logging.CRITICAL)
    ch.setFormatter(logging.Formatter('%(levelname)s (%(name)s): %(message)s'))
    root_logger.addHandler(ch)

    # Pipelines Definition
    pipes = define_pipeline.parse_steps([config.step0, config.step1,
                                             config.step2, config.step3])

    # Pipelines Evaluation
    outFolder = pipelines.run(pipes=pipes, X=X, exp_tag=fileName, root=root, y=y)

    # Copy the ade_config just used into the outFolder
    shutil.copy(config_path, os.path.join(outFolder, 'ade_config.py'))
    # Move the logging file into the outFolder
    shutil.move(logFileName, outFolder)

# ----------------------------  RUN MAIN ---------------------------- #
if __name__ == '__main__':
    from adenine import __version__
    parser = argparse.ArgumentParser(#usage="%(prog)s [-c] configuration_file.py",
                                     description='Adenine script for generating pipelines.')
    parser.add_argument('--version', action='version', version='%(prog)s v'+__version__)
    parser.add_argument("-c", "--create", dest="create", action="store_true",
                        help="create config file", default=False)
    parser.add_argument("configuration_file", help="specify config file", default='ade_config.py')
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
