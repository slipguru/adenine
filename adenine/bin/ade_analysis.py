#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys, os
import logging
import cPickle as pkl

def main(dumpfile):
    
    # Load the results
    with open(dumpfile, 'r') as f:
        res = pkl.load(f)
        
    print res

    


# ----------------------------  RUN MAIN ---------------------------- #
if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print("USAGE: ade_analysis.py <RESULTS_FOLDER> ")
        sys.exit(-1)
    else:
        fileNames = [ f for f in os.listdir(sys.argv[1]) if os.path.isfile(os.path.join(sys.argv[1],f)) ]
        found = False
        for f in fileNames:
            if f.endswith('.pkl'):
                found, fileName = True, f
                break
    
        if not found:
            print("No .pkl file found in {}".format(sys.argv[1]))
            sys.exit(-1)
        else:
            print("Starting the analysis of {}".format(fileName))
            main(os.path.join(sys.argv[1],fileName)) # Run analysis
                