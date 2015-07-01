#!/usr/bin/python
# -*- coding: utf-8 -*-

import imp, sys, os
import logging
from adenine.core import define_pipeline
from adenine.core import pipelines


def main(config_file):
    
    # Define the ade.log file (a new one for each run)
    logFileName = 'ade.log'
    logging.basicConfig(filename=logFileName, level=logging.INFO, filemode = 'w')

    # Load the configuration file
    config_path = os.path.abspath(config_file)
    config = imp.load_source('ade_config', config_path)
    
    # Get the experiment tag and the output root folder
    exp_tag, root = config.exp_tag, config.output_root_folder

    # Read the variables from the config file
    X, y, feat_names = config.X, config.y, config.feat_names

    # Pipelines Definition
    pipes_def = define_pipeline.parse_steps([config.step0, config.step1,
                                             config.step2, config.step3])
    
    # Pipelines Creation
    pipes = pipelines.create(pipes_def)
    
    # Pipelines Evaluation
    pipelines.run(pipes, X, y, feat_names, exp_tag, root)
    
    # print("See {} for details".format(logFileName))


# ----------------------------  RUN MAIN ---------------------------- #
if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("USAGE: ade_run.py <CONFIG_FILE> ")
        sys.exit()
    else:
        main(sys.argv[1])
        
    
