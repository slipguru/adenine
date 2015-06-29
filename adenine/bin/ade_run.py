#!/usr/bin/python
# -*- coding: utf-8 -*-

import imp, sys, os


def main(config_file):

    # Load the configuration file
    config_path = os.path.abspath(config_file)
    config = imp.load_source('ade_config', config_path)

    # Read the variables from the config file
    X, y = config.X, config.y

    print X
    print y

    







# ----------------------------  RUN MAIN ---------------------------- #
if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("USAGE: ade_run.py <CONFIG_FILE> ")
        sys.exit()
    else:
        out = main(sys.argv[1])
