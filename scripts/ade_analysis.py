#!/usr/bin/env python
# -*- coding: utf-8 -*-

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Federico Tomasi, Annalisa Barla
#
# FreeBSD License
######################################################################

from __future__ import print_function

import imp, sys, os
import time
import logging
import cPickle as pkl
import gzip
import argparse

from adenine.core import analyze_results
from adenine.utils import extra

def main(dumpfile):
    # Load the configuration file
    config_path = os.path.dirname(dumpfile)
    config_path = os.path.join(os.path.abspath(config_path), 'ade_config.py')
    config = imp.load_source('ade_config', config_path)
    extra.set_module_defaults(config, {'file_format': 'pdf',
                                       'plotting_context': 'paper'})

    # Read the variables from the config file
    feat_names, class_names = config.feat_names, config.class_names
    # Load the results used with ade_run.py
    try:
        with gzip.open(os.path.join(os.path.dirname(dumpfile),'__data.pkl.tz'), 'r') as f:
            data = pkl.load(f)
            X, y = data['X'], data['y']
    except:
        sys.stderr("Cannot load __data.pkl.tz. Reloading data from config file ...")
        X, y = config.X, config.y

    # Initialize the log file
    fileName = 'results_'+os.path.basename(dumpfile)[0:-7]
    logFileName = os.path.join(os.path.dirname(dumpfile), fileName+'.log')
    logging.basicConfig(filename=logFileName, level=logging.INFO, filemode='w',
                        format='%(levelname)s (%(name)s): %(message)s')
    root_logger = logging.getLogger()
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    ch.setFormatter(logging.Formatter('%(levelname)s (%(name)s): %(message)s'))
    root_logger.addHandler(ch)

    tic = time.time()
    print("\nUnpickling output ...", end=' ')
    # Load the results
    with gzip.open(dumpfile, 'r') as f:
        res = pkl.load(f)

    print("done: {} s".format(extra.sec_to_time(time.time()-tic)))

    # Analyze the pipelines
    analyze_results.analyze(input_dict=res, root=os.path.dirname(dumpfile),
                            y=y, feat_names=feat_names, class_names=class_names,
                            plotting_context=config.plotting_context,
                            file_format=config.file_format)

# ----------------------------  RUN MAIN ---------------------------- #
if __name__ == '__main__':
    from adenine import __version__
    parser = argparse.ArgumentParser(#usage="%(prog)s RESULTS_DIR",
                                     description='Adenine script for analysing pipelines.')
    parser.add_argument('--version', action='version', version='%(prog)s v'+__version__)
    parser.add_argument("result_folder", help="specify results directory")
    args = parser.parse_args()
    root_folder = args.result_folder
    filename = [f for f in os.listdir(root_folder) if os.path.isfile(os.path.join(root_folder, f)) and f.endswith('.pkl.tz') and f !=  "__data.pkl.tz"]
    if not filename:
        sys.stderr.write("No .pkl file found in {}. Aborting...".format(root_folder))
        sys.exit(-1)

    # print("Starting the analysis of {}".format(filename))
    main(os.path.join(os.path.abspath(root_folder), filename[0])) # Run analysis
