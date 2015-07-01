#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import logging
import cPickle as pkl
from adenine.utils.extra import make_time_flag
from sklearn.pipeline import Pipeline

def create(pdef):
    """Scikit-learn Pipelines objects creation.
    
    This function creates a list of sklearn Pipeline objects starting from the list of list of tuples given in input that could be created using the adenine.core.define_pipeline module.
    
    Parameters
    -----------
    pdef : list of list of tuples
        This arguments contains the specification needed by sklearn in order to create a working Pipeline object.
    
    Returns
    -----------
    pipes : list of sklearn.pipeline.Pipeline objects
        The list of Piplines, each of them can be fitted and trasformed with some data.
    """
    pipes = []
    for p in pdef:
        pipes.append(Pipeline(p))
        
    return pipes
    
    
def run(pipes, X, y = [], feat_names = [], exp_tag = 'def_tag', root = ''):
    """Fit and transform some pipelines on some data.
    
    This function fits each pipeline in the input list against the provided data. The results are dumped into a pkl file.
    
    Parameters
    -----------
    pdef : list of list of tuples, default : 'Default'
        This arguments contains the specification needed by sklearn in order to create a working Pipeline object.
    
    Returns
    -----------
    pipes : list of sklearn.pipeline.Pipeline objects
        The list of Piplines, each of them can be fitted and trasformed with some data.
    """
    # Check root folder
    if not os.path.exists(root): # if it does not exist
        if not len(root):             # (and the name has not been even specified)
            root = 'results_'+exp_tag+make_time_flag() # create a standard name
        os.makedirs(root)        # and make the folder
        logging.warn("No root folder supplied, folder {} created".format(os.path.abspath(root)))
        
    # Fit
    for p in [pipes[0]]:
        p.set_params(KMeans__n_clusters = 4)
        o = p.fit(X,y)
        print p.steps[1]
        
        c = p.steps[1].transform(X)
        
        import matplotlib.pyplot as plt
        plt.scatter(c[:,0], c[:,1], s = 100, c = y, alpha = 0.5)
        plt.show()
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        