#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module is just a wrapper for some sklearn.datasets functions."""

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Federico Tomasi, Annalisa Barla
#
# FreeBSD License
######################################################################

import numpy as np
import pandas as pd
import logging
from sklearn import datasets
from sklearn.preprocessing import Binarizer

# Legacy import
try:
    from sklearn.model_selection import StratifiedShuffleSplit
except ImportError:
    from sklearn.cross_validation import StratifiedShuffleSplit


def generate_gauss(mu=None, std=None, n_sample=None):
    """Create a Gaussian dataset.

    Generates a dataset with n_sample * n_class examples and n_dim dimensions.

    Parameters
    -----------
    mu : array of float, shape : n_class x n_dim
        The mean of each class.

    std :  array of float, shape : n_class
        The standard deviation of each Gaussian distribution.

    n_sample : int
        Number of point per class.
    """
    n_class, n_var = mu.shape

    X = np.zeros((n_sample*n_class, n_var))
    y = np.zeros(n_sample*n_class, dtype=int)

    start = 0
    for i, s, m in zip(range(n_class), std, mu):
        end = start + n_sample
        X[start:end, :] = s * np.random.randn(n_sample, n_var) + m
        y[start:end] = i
        start = end

    return X, y


def load_custom(x_filename, y_filename, samples_on='rows', **kwargs):
    """Load a custom dataset.

    This function loads the data matrix and the label vector returning a
    unique sklearn-like object dataSetObj.

    Parameters
    -----------
    x_filename : string
        The data matrix file name.

    y_filename : string
        The label vector file name.

    samples_on : string
        This can be either in ['row', 'rows'] if the samples lie on the row of
        the input data matrix, or viceversa in ['col', 'cols'] the other way
        around.

    data_sep : string
        The data separator. For instance comma, tab, blank space, etc.

    Returns
    -----------
    data : sklearn.datasets.base.Bunch
        An instance of the sklearn.datasets.base.Bunch class, the meaningful
        attributes are .data, the data matrix, and .target, the label vector.
    """
    if x_filename is None:
        raise IOError("Filename for X must be specified with mode 'custom'.")

    if x_filename.endswith('.npy'):  # it an .npy file is provided
        try:  # labels are not mandatory
            y = np.load(y_filename)
        except IOError as e:
            y = None
            e.strerror = "No labels file provided"
            logging.error("I/O error({0}): {1}".format(e.errno, e.strerror))
        X = np.load(x_filename)
        if samples_on not in ['row', 'rows']:
            # data matrix must be n_samples x n_features
            X = X.T
        return datasets.base.Bunch(data=X, target=y,
                                   index=np.arange(X.shape[0]))

    elif x_filename.endswith('.csv') or x_filename.endswith('.txt'):
        y = None
        try:
            dfx = pd.io.parsers.read_csv(x_filename, **kwargs)
            if samples_on not in ['row', 'rows']:
                # data matrix must be n_samples x n_features
                dfx = dfx.transpose()
            if y_filename is not None:
                # Before loading labels, remove parameters that were likely
                # specified for data only.
                kwargs.pop('usecols', None)
                y = pd.io.parsers.read_csv(y_filename,
                                           **kwargs).as_matrix().ravel()
        except IOError as e:
            e.strerror = "Can't open {} or {}".format(x_filename, y_filename)
            logging.error("I/O error({0}): {1}".format(e.errno, e.strerror))

        return datasets.base.Bunch(data=dfx.as_matrix(), feature_names=dfx.columns.tolist(),
                                   target=y, index=dfx.index.tolist())


def load(opt='custom', x_filename=None, y_filename=None, n_samples=0,
         samples_on='rows', **kwargs):
    """Load a specified dataset.

    This function can be used either to load one of the standard scikit-learn
    datasets or a different dataset saved as X.npy Y.npy in the working
    directory.

    Parameters
    -----------
    opt : {'iris', 'digits', 'diabetes', 'boston', 'circles', 'moons',
          'custom'}, default: 'custom'
        Name of a predefined dataset to be loaded.

    x_filename : string, default : None
        The data matrix file name.

    y_filename : string, default : None
        The label vector file name.

    n_samples : int
        The number of samples to be loaded. This comes handy when dealing with
        large datasets. When n_samples is less than the actual size of the
        dataset this function performs a random subsampling that is stratified
        w.r.t. the labels (if provided).

    samples_on : string
        This can be either in ['row', 'rows'] if the samples lie on the row of
        the input data matrix, or viceversa in ['col', 'cols'] the other way
        around.

    data_sep : string
        The data separator. For instance comma, tab, blank space, etc.

    Returns
    -----------
    X : array of float, shape : n_samples x n_features
        The input data matrix.

    y : array of float, shape : n_samples
        The label vector; np.nan if missing.

    feature_names : array of integers (or strings), shape : n_features
        The feature names; a range of number if missing.

    index : list of integers (or strings)
        This is the samples identifier, if provided as first column (or row) of
        of the input file. Otherwise it is just an incremental range of size
        n_samples.
    """
    data = None
    try:
        if opt.lower() == 'iris':
            data = datasets.load_iris()
        elif opt.lower() == 'digits':
            data = datasets.load_digits()
        elif opt.lower() == 'diabetes':
            data = datasets.load_diabetes()
            b = Binarizer(threshold=np.mean(data.target))
            data.target = b.fit_transform(data.data)
        elif opt.lower() == 'boston':
            data = datasets.load_boston()
            b = Binarizer(threshold=np.mean(data.target))
            data.target = b.fit_transform(data.data)
        elif opt.lower() == 'gauss':
            means = np.array([[-1, 1, 1, 1], [0, -1, 0, 0], [1, 1, -1, -1]])
            sigmas = np.array([0.33, 0.33, 0.33])
            if n_samples <= 1:
                n_samples = 333
            xx, yy = generate_gauss(mu=means, std=sigmas, n_sample=n_samples)
            data = datasets.base.Bunch(data=xx, target=yy)
        elif opt.lower() == 'circles':
            if n_samples == 0:
                n_samples = 400
            xx, yy = datasets.make_circles(n_samples=n_samples, factor=.3,
                                           noise=.05)
            data = datasets.base.Bunch(data=xx, target=yy)
        elif opt.lower() == 'moons':
            if n_samples == 0:
                n_samples = 400
            xx, yy = datasets.make_moons(n_samples=n_samples, noise=.01)
            data = datasets.base.Bunch(data=xx, target=yy)
        elif opt.lower() == 'custom':
            data = load_custom(x_filename, y_filename, samples_on, **kwargs)
    except IOError as e:
        print("I/O error({0}): {1}".format(e.errno, e.strerror))

    X, y = data.data, data.target
    if n_samples > 0 and X.shape[0] > n_samples:
        if y is not None:
            try:  # Legacy for sklearn
                sss = StratifiedShuffleSplit(y, test_size=n_samples, n_iter=1)
                # idx = np.random.permutation(X.shape[0])[:n_samples]
            except TypeError:
                sss = StratifiedShuffleSplit(n_iter=1, test_size=n_samples) \
                    .split(X, y)
            _, idx = list(sss)[0]
        else:
            idx = np.arange(X.shape[0])
            np.random.shuffle(idx)
            idx = idx[:n_samples]

        X, y = X[idx, :], y[idx]

    feat_names = data.feature_names if hasattr(data, 'feature_names') \
        else np.arange(X.shape[1])
    index = data.index if hasattr(data, 'index') \
        else np.arange(X.shape[0])

    return X, y, feat_names, index
