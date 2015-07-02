#!/usr/bin/python
# -*- coding: utf-8 -*-

import imp, sys, os
import logging
from adenine.utils.extra import make_time_flag
from adenine.core import define_pipeline
from adenine.core import pipelines


def main(config_file):
    
    # Load the configuration file
    config_path = os.path.abspath(config_file)
    config = imp.load_source('ade_config', config_path)
    
    # Get the experiment tag and the output root folder
    exp_tag, root = config.exp_tag, config.output_root_folder
    
    # Define the ade.log file (a new one for each run)
    fileName = 'ade_'+exp_tag+'_'+make_time_flag()
    logFileName = os.path.join(root, fileName+'.log')
    logging.basicConfig(filename=logFileName, level=logging.INFO, filemode = 'w')

    # Read the variables from the config file
    X, y, feat_names = config.X, config.y, config.feat_names

    # Pipelines Definition
    pipes = define_pipeline.parse_steps([config.step0, config.step1,
                                             config.step2, config.step3])
    
    # Pipelines Evaluation
    pipelines.run(pipes = pipes, X = X, exp_tag = fileName, root = root, parallel = config.parallel)
    
    # print("See {} for details".format(logFileName))


# ----------------------------  RUN MAIN ---------------------------- #
if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("USAGE: ade_run.py <CONFIG_FILE> ")
        sys.exit()
    else:
        main(sys.argv[1])
        
    
