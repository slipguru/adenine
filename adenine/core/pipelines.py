#!/usr/bin/python
# -*- coding: utf-8 -*-

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Federico Tomasi, Annalisa Barla
#
# FreeBSD License
######################################################################

import copy
import logging
import numpy as np


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
    return [Pipeline(p) for p in pdef]


def which_level(label):
    """Define the step level according to the input step label [DEPRECATED].

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
    if not isinstance(label, basestring):
        raise ValueError("String expected")

    label = label.lower()
    if label.startswith('impute'):
        level = 'imputing'
    elif label in ('recenter', 'standardize', 'normalize', 'minmax'):
        level = 'preproc'
    elif label in ('pca', 'incrementalpca', 'randomizedpca', 'kernelpca',
                   'isomap', 'lle', 'se', 'mds', 'tsne', 'rbm'):
        level = 'dimred'
    elif label in ('kmeans', 'ap', 'ms', 'spectral',
                   'hierarchical'):
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
    if level in ('imputing', 'preproc', 'dimred', 'None'):
        if hasattr(step, 'embedding_'):
            res = step.embedding_
        else:
            res = step.transform(X)
    elif level == 'clustering':
        if hasattr(step, 'labels_'):
            res = step.labels_  # e.g. in case of spectral clustering
        elif hasattr(step, 'affinity') and step.affinity == 'precomputed':
            if not hasattr(step.estimator, 'labels_'):
                step.estimator.fit(X)
            res = step.estimator.labels_
        else:
            res = step.predict(X)
    return res


def pipe_worker(pipe_id, pipe, pipes_dump, X):
    """Parallel pipelines execution.

    Parameters
    -----------
    pipe_id : string
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
        step_id = 'step' + str(j)
        # 1. define which level of step is this (i.e.: imputing, preproc,
        # dimred, clustering, none)
        level = step[-1]
        # 2. fit the model (whatever it is)
        if step[1].get_params().get('method') == 'hessian':
            # check hessian lle constraints
            n_components = step[1].get_params().get('n_components')
            n_neighbors = 1 + (n_components * (n_components + 3) / 2)
            step[1].set_params(n_neighbors=n_neighbors)
        try:
            step[1].fit(X_curr)

            # 3. evaluate (i.e. transform or predict according to the level)
            # X_curr = evaluate(level, step[1], X_curr)
            X_next = evaluate(level, step[1], X_curr)
            # 3.1 if the model is suitable for voronoi tessellation: fit also
            # on 2D
            mdl_voronoi = None
            if hasattr(step[1], 'cluster_centers_'):
                mdl_voronoi = copy.copy(step[1].best_estimator_ if hasattr(
                    step[1], 'best_estimator_') else step[1])
                if not hasattr(step[1], 'affinity') or step[1].affinity != 'precomputed':
                    mdl_voronoi.fit(X_curr[:, :2])
                else:
                    mdl_voronoi.fit(X_curr)

            # 4. save the results in a dictionary of dictionaries of the form:
            # save memory and do not dump data after preprocessing (unused in
            # analysys)
            if level in ('preproc', 'imputing'):
                result = [step[0], level, step[1].get_params(),
                          np.empty(0), np.empty(0), step[1], mdl_voronoi]
                X_curr = np.array(X_next)  # update the matrix

            # save memory dumping X_curr only in case of clustering
            elif level == 'dimred':
                result = [step[0], level, step[1].get_params(),
                          X_next, np.empty(0), step[1], mdl_voronoi]
                X_curr = X_next  # update the matrix

            # clustering
            elif level == 'clustering':
                result = [step[0], level, step[1].get_params(),
                          X_next, X_curr, step[1], mdl_voronoi]
            if level != 'None':
                step_dump[step_id] = result

        except (AssertionError, ValueError) as e:
            logging.critical("Pipeline %s failed at step %s. "
                             "Traceback: %s", pipe_id, step[0], e)


    # Monkey-patch, see: https://github.com/scikit-learn/scikit-learn/issues/7562
    # and wait for the next numpy update
    # step_dump['step2'][-2] = None

    if pipes_dump is None:
        return step_dump

    pipes_dump[pipe_id] = step_dump
