#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import logging
import cPickle as pkl
import numpy as np
from adenine.utils.extra import make_time_flag

def create(pdef):
    """Scikit-learn Pipelines objects creation (deprecated).
    
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
    from sklearn.pipeline import Pipeline
    pipes = []
    for p in pdef:
        pipes.append(Pipeline(p))
    return pipes
    
def which_level(label):
    """Define the step level according to the input step label.
    
    This function return the level (i.e.: imputing, preproc, dimred, clustring, None) according to the step label provided as input.
    
    Parameters
    -----------
    label : string
        This is the step level as it is reported in the ade_config file.
    
    Returns
    -----------
    level : {imputing, preproc, dimred, clustring, None}
        The appropriate level of the input step.
    """
    if label in set(['Impute_median', 'Impute_mean']):
        level = 'imputing'
    elif label in set(['Recenter', 'Standardize', 'Normalize', 'MinMax']):
        level = 'preproc'
    elif label in set(['PCA', 'IncrementalPCA', 'RandomizedPCA', 'KernelPCA', 'Isomap','LLE', 'SE', 'MDS', 'tSNE']):
        level = 'dimred'
    elif label in set(['KMeans', 'KernelKMeans', 'AP', 'MS', 'Spectral', 'Hierarchical']):
        level = 'clustering'
    else:
        level = 'None'
    return level
    
def evaluate(level, step, X):
    """Transform or predict according to the input level.
    
    This function uses the transform or the predict method on the input sklearn-like step according to its level (i.e. imputing, preproc, dimred, clustering, none).
    
    Parameters
    -----------
    level : {'imputing', 'preproc', 'dimred', 'clustering', 'None'}
        The step level.
    
    step : sklearn-like object
        This might be an Imputer, or a PCA, or a KMeans (and so on...) sklearn-like object.
    
    X : array of float, shape : n_samples x n_features
        The input data matrix.
    
    Returns
    -----------
    res : array of float
        A matrix projection in case of dimred, a label vector in case of clustering, and so on.
    """
    if level == 'imputing' or level == 'preproc' or level == 'dimred' or level == 'None':
        if hasattr(step, 'embedding_'):
            res = step.embedding_
        else:
            res = step.transform(X)
    elif level == 'clustering':
        if hasattr(step, 'labels_'):
            res = step.labels_ # e.g. in case of spectral clustering
        else:    
            res = step.predict(X)
    return res
    
def run(pipes = (), X = (), exp_tag = 'def_tag', root = ''):
    """Fit and transform/predict some pipelines on some data.
    
    This function fits each pipeline in the input list on the provided data. The results are dumped into a pkl file as a dictionary of dictionaries of the form {'pipeID': {'stepID' : [alg_name, level, params, data_out, data_in, model_obj], ...}, ...}. If a pipeline fails for some reasons the content of the stepID key is a list of np.nan.
    
    Parameters
    -----------
    pipes : list of list of tuples
        Each tuple contains a label and a sklearn Pipeline object.
        
    X : array of float, shape : n_samples x n_features, default : ()
        The input data matrix.
        
    exp_tag : string
        An intuitive tag for the current experiment.
    
    root : string
        The root folder to save the results.
        
    parallel : bool
        Run all the pipelines in parallel on the cores of your machine (if True, it requires pplus).
        
    Returns
    -----------
    outputFolderName : string
        The path of the output folder.
    """
    # Check root folder
    if not os.path.exists(root): # if it does not exist
        if not len(root):             # (and the name has not been even specified)
            root = 'results_'+exp_tag+make_time_flag() # create a standard name
        os.makedirs(root)        # and make the folder
        logging.warn("No root folder supplied, folder {} created".format(os.path.abspath(root)))
        
    # Eval pipes
    pipes_dump = dict()
    for i, pipe in enumerate(pipes):
        pipeID = 'pipe'+str(i)
        step_dump = dict()
        X_curr = np.array(X) # COPY X as X_curr (to avoid that the next pipeline works on the results of the previuos one)
        for j, step in enumerate(pipe): # step[0] -> step_label | step[1] -> sklearn (or sklearn-like) object (model)
            stepID = 'step'+str(j)
            # 1. define which level of step is this (i.e.: imputing, preproc, dimred, clustering, none)
            level = which_level(step[0])
            # 2. fit the model (whatever it is)
            if step[1].get_params().get('method') == 'hessian': # check hessian lle constraints
                n_components = step[1].get_params().get('n_components')
                step[1].set_params(n_neighbors = 1 + (n_components * (n_components + 3) / 2))
            try:
                step[1].fit(X_curr)
                # 3. evaluate (i.e. transform or predict according to the level)
                # X_curr = evaluate(level, step[1], X_curr)
                X_next = evaluate(level, step[1], X_curr)
                # 4. save the results in a dictionary of dictionary of the form:
                # {'pipeID': {'stepID' : [alg_name, level, params, res]}}
                step_dump[stepID] = [step[0], level, step[1].get_params(), X_next, X_curr, step[1]]
                X_curr = np.array(X_next) # update the matrix
            except AssertionError as e:
                logging.critical("Pipeline {} failed at step {}".format(pipeID, step[0]))
                step_dump[stepID] = [step[0], level, step[1].get_params(), np.nan, np.nan, np.nan]
                break # skip this pipeline
        pipes_dump[pipeID] = step_dump
        logging.debug("DUMP: \n {} \n #########".format(pipes_dump))

    # Output Name
    outputFileName = exp_tag
    outputFolderName = os.path.join(root,outputFileName)

    # Create exp folder into the root folder
    os.makedirs(outputFolderName)

    # pkl Dump
    with open(os.path.join(outputFolderName,outputFileName+'.pkl'), 'w+') as f:
        pkl.dump(pipes_dump, f)
    logging.info("Dumped : {}".format(os.path.join(outputFolderName,outputFileName+'.pkl')))
    
    return outputFolderName
            
    
            
            
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        