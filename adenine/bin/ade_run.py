#!/usr/bin/ipython
# -*- coding: utf-8 -*-

import imp, sys, os, shutil
import time
import logging
from adenine.utils.extra import make_time_flag
from adenine.core import define_pipeline
from adenine.core import pipelines

def main(config_file):

    # Load the configuration file
    config_path = os.path.abspath(config_file)
    config = imp.load_source('ade_config', config_path)

    # Read the variables from the config file
    X, y, feat_names, class_names = config.X, config.y, config.feat_names, config.class_names

    # Get the experiment tag and the output root folder
    exp_tag, root = config.exp_tag, config.output_root_folder
    if not os.path.exists(root):
        os.makedirs(root)

    # Define the ade.log file (a new one for each run)
    fileName = 'ade_'+exp_tag+'_'+make_time_flag()
    logFileName = os.path.join(root, fileName+'.log')
    logging.basicConfig(filename=logFileName, level=logging.INFO, filemode = 'w')

    # Pipelines Definition
    pipes = define_pipeline.parse_steps([config.step0, config.step1,
                                             config.step2, config.step3])

    # Pipelines Evaluation
    tic = time.time()
    outFolder = pipelines.run(pipes = pipes, X = X, exp_tag = fileName, root = root)
    tac = time.time()

    # Copy the ade_config just used into the outFolder
    shutil.copy(config_path, os.path.join(outFolder, 'ade_config.py'))
    # Move the logging file into the outFolder
    shutil.move(logFileName, outFolder)
    print("\n\npipelines.run: Elapsed time : {}".format(tac-tic))

# ----------------------------  RUN MAIN ---------------------------- #
if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("USAGE: ade_run.py <CONFIG_FILE> ")
        sys.exit(-1)
    else:
        main(sys.argv[1])
