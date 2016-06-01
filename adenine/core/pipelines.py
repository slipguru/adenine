#!/usr/bin/python
# -*- coding: utf-8 -*-

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Federico Tomasi, Annalisa Barla
#
# FreeBSD License
######################################################################

import os
import copy
import logging
import multiprocessing
import cPickle as pkl
import numpy as np
# from adenine.utils.extra import make_time_flag
from adenine.utils.extra import get_time
from adenine.utils.extra import timed


def create(pdef):
    """Scikit-learn Pipelines objects creation (deprecated).

    This function creates a list of sklearn Pipeline objects starting from the
    list of list of tuples given in input that could be created using the
    adenine.core.define_pipeline module.

    Parameters
    -----------
    pdef : list of list of tuples
        This arguments contains the specification needed by sklearn in order
        to create a working Pipeline object.

    Returns
    -----------
    pipes : list of sklearn.pipeline.Pipeline objects
        The list of Piplines, each of them can be fitted and trasformed
        with some data.
    """
    from sklearn.pipeline import Pipeline
    pipes = []
    for p in pdef:
        pipes.append(Pipeline(p))
    return pipes


def which_level(label):
    """Define the step level according to the input step label.

    This function return the level (i.e.: imputing, preproc, dimred, clustring,
    None) according to the step label provided as input.

    Parameters
    -----------
    label : string
        This is the step level as it is reported in the ade_config file.

    Returns
    -----------
    level : {imputing, preproc, dimred, clustering, None}
        The appropriate level of the input step.
    """
    if label in ('Impute_median', 'Impute_mean', 'Impute'):
        level = 'imputing'
    elif label in ('Recenter', 'Standardize', 'Normalize', 'MinMax'):
        level = 'preproc'
    elif label in ('PCA', 'IncrementalPCA', 'RandomizedPCA', 'KernelPCA',
                   'Isomap', 'LLE', 'SE', 'MDS', 'tSNE'):
        level = 'dimred'
    elif label in ('KMeans', 'KernelKMeans', 'AP', 'MS', 'Spectral',
                   'Hierarchical'):
        level = 'clustering'
    else:
        level = 'None'
    return level


def evaluate(level, step, X):
    """Transform or predict according to the input level.

    This function uses the transform or the predict method on the input
    sklearn-like step according to its level (i.e. imputing, preproc, dimred,
    clustering, none).

    Parameters
    -----------
    level : {'imputing', 'preproc', 'dimred', 'clustering', 'None'}
        The step level.

    step : sklearn-like object
        This might be an Imputer, or a PCA, or a KMeans (and so on...)
        sklearn-like object.

    X : array of float, shape : n_samples x n_features
        The input data matrix.

    Returns
    -----------
    res : array of float
        A matrix projection in case of dimred, a label vector in case of
        clustering, and so on.
    """
    if level in ['imputing', 'preproc', 'dimred', 'None']:
        if hasattr(step, 'embedding_'):
            res = step.embedding_
        else:
            res = step.transform(X)
    elif level == 'clustering':
        if hasattr(step, 'labels_'):
            res = step.labels_  # e.g. in case of spectral clustering
        else:
            res = step.predict(X)
    return res


def pipe_worker(pipeID, pipe, pipes_dump, X):
    """Parallel pipelines execution.

    Parameters
    -----------
    pipeID : string
        Pipeline identifier.

    pipe : list of tuples
        Tuple containing a label and a sklearn Pipeline object.

    pipes_dump : multiprocessing.Manager.dict
        Dictionary containing the results of the parallel execution.

    X : array of float, shape : n_samples x n_features, default : ()
        The input data matrix.
    """
    step_dump = dict()

    # COPY X as X_curr (to avoid that the next pipeline
    # works on the results of the previuos one)
    X_curr = np.array(X)
    for j, step in enumerate(pipe):
        # step[0] -> step_label | step[1] -> model, sklearn (or sklearn-like)
        # object
        stepID = 'step'+str(j)
        # 1. define which level of step is this (i.e.: imputing, preproc,
        # dimred, clustering, none)
        level = which_level(step[0])
        # 2. fit the model (whatever it is)
        if step[1].get_params().get('method') == 'hessian':
            # check hessian lle constraints
            n_components = step[1].get_params().get('n_components')
            step[1].set_params(n_neighbors=1+(n_components*(n_components+3)/2))
        try:
            step[1].fit(X_curr)

            # 3. evaluate (i.e. transform or predict according to the level)
            # X_curr = evaluate(level, step[1], X_curr)
            X_next = evaluate(level, step[1], X_curr)
            # 3.1 if the model is suitable for voronoi tessellation: fit also
            # on 2D
            mdl_voronoi = None
            if hasattr(step[1], 'cluster_centers_'):
                if hasattr(step[1], 'best_estimator_'):
                    mdl_voronoi = copy.copy(step[1].best_estimator_)
                else:
                    mdl_voronoi = copy.copy(step[1])
                mdl_voronoi.fit(X_curr[:, :2])

            # 4. save the results in a dictionary of dictionaries of the form:
            # {'pipeID': {'stepID' : [alg_name, level, params, res, Xnext, Xcurr, stepObj, voronoi_suitable_model]}}
            if level in ('preproc', 'imputing'):  # save memory and do not dump data after preprocessing (not used in analysys)
                step_dump[stepID] = [step[0], level, step[1].get_params(), np.array([]), np.array([]), step[1], mdl_voronoi]
                X_curr = np.array(X_next)  # update the matrix
            elif level == 'dimred':  # save memory dumping X_curr only in case of clusutering
                step_dump[stepID] = [step[0], level, step[1].get_params(), X_next, np.array([]), step[1], mdl_voronoi]
                X_curr = X_next  # update the matrix
            elif level == 'clustering':  # clustering
                step_dump[stepID] = [step[0], level, step[1].get_params(), X_next, X_curr, step[1], mdl_voronoi]

            # if level != 'clustering':
            #     X_curr = np.array(X_next) # update the matrix

        except (AssertionError, ValueError) as e:
            logging.critical("Pipeline {} failed at step {}. "
                             "Traceback: {}".format(pipeID, step[0], e))
            step_dump[stepID] = [step[0], level, step[1].get_params(), np.nan,
                                 np.nan, np.nan]

    pipes_dump[pipeID] = step_dump
    logging.debug("DUMP: \n {} \n #########".format(pipes_dump))


@timed
def run(pipes=(), X=(), exp_tag='def_tag', root='', y=None):
    """Fit and transform/predict some pipelines on some data.

    This function fits each pipeline in the input list on the provided data.
    The results are dumped into a pkl file as a dictionary of dictionaries of
    the form {'pipeID': {'stepID' : [alg_name, level, params, data_out,
    data_in, model_obj, voronoi_suitable_object], ...}, ...}. The model_obj is
    the sklearn model which has been fit on the dataset, the
    voronoi_suitable_object is the very same model but fitted on just the first
    two dimensions of the dataset. If a pipeline fails for some reasons the
    content of the stepID key is a list of np.nan.

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

    y : array-like, optional
        If specified, it contains data labels.

    Returns
    -----------
    outputFolderName : string
        The path of the output folder.
    """
    # Check root folder
    if not os.path.exists(root):  # if it does not exist
        if not len(root):         # (and the name has not been even specified)
            root = 'results_'+exp_tag+get_time()  # create a standard name
        os.makedirs(root)         # and make the folder
        logging.warn("No root folder supplied, "
                     "folder {} created".format(os.path.abspath(root)))

    jobs = []
    manager = multiprocessing.Manager()
    pipes_dump = manager.dict()

    # Submit jobs
    for i, pipe in enumerate(pipes):
        pipeID = 'pipe'+str(i)
        p = multiprocessing.Process(target=pipe_worker,
                                    args=(pipeID, pipe, pipes_dump, X))
        jobs.append(p)
        p.start()
        logging.info("Job: {} submitted".format(pipeID))

    # Collect results
    ret_count = 0
    for proc in jobs:
        proc.join()
        ret_count += 1
    logging.info("{} jobs collected".format(ret_count))

    # Convert the DictProxy to standard dict
    pipes_dump = dict(pipes_dump)

    # Output Name
    output_filename = exp_tag
    output_folder = os.path.join(root, output_filename)

    # Create exp folder into the root folder
    os.makedirs(output_folder)

    # pkl Dump
    import gzip
    with gzip.open(os.path.join(output_folder,
                                output_filename+'.pkl.tz'), 'w+') as f:
        pkl.dump(pipes_dump, f)
    logging.info("Dumped : {}".format(os.path.join(output_folder,
                                                   output_filename+'.pkl.tz')))
    with gzip.open(os.path.join(output_folder, '__data.pkl.tz'), 'w+') as f:
        pkl.dump({'X': X, 'y': y}, f)
    logging.info("Dumped : {}".format(os.path.join(output_folder,
                                                   '__data.pkl.tz')))

    return output_folder
