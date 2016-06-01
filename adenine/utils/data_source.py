#!/usr/bin/python
# -*- coding: utf-8 -*-

"""This module is just a wrapper for some sklearn.datasets functions"""

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Federico Tomasi, Annalisa Barla
#
# FreeBSD License
######################################################################

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import Binarizer

# Legacy import
try:
    from sklearn.model_selection import StratifiedShuffleSplit
except ImportError:
    from sklearn.cross_validation import StratifiedShuffleSplit


def generate_gauss(mu=(), std=(), n_sample=()):
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


def load_custom(x_filename, y_filename):
    """Load a custom dataset.

    This function loads the data matrix and the label vector returning a
    unique sklearn-like object dataSetObj.

    Parameters
    -----------
    x_filename : string
        The data matrix file name.

    y_filename : string
        The label vector file name.

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
            y = np.nan
            e.strerror = "No labels file provided"
            print("I/O error({0}): {1}".format(e.errno, e.strerror))

        return datasets.base.Bunch(data=np.load(x_filename), target=y)
        # return dataSetObj(np.load(x_filename),y)

    elif x_filename.endswith('.csv') or x_filename.endswith('.txt'):
        try:
            dfx = pd.io.parsers.read_csv(x_filename, header=0, index_col=0)
        except IOError as e:
            e.strerror = "Can't open {}".format(x_filename)
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
        y = None
        if y_filename is not None:
            y = pd.io.parsers.read_csv(y_filename,
                                       header=0,
                                       index_col=0).as_matrix().ravel()
        return datasets.base.Bunch(data=dfx.as_matrix(), target=y)


def load(opt='custom', x_filename=None, y_filename=None, n_samples=0):
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

    Returns
    -----------
    X : array of float, shape : n_samples x n_features
        The input data matrix.

    y : array of float, shape : n_samples
        The label vector; np.nan if missing.

    feature_names : array of integers (or strings), shape : n_features
        The feature names; a range of number if missing.
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
            if n_samples <= 1: n_samples = 333
            xx, yy = generate_gauss(mu=means, std=sigmas, n_sample=n_samples)
            data = datasets.base.Bunch(data=xx, target=yy)
        elif opt.lower() == 'circles':
            if n_samples == 0: n_samples = 400
            xx, yy = datasets.make_circles(n_samples=n_samples, factor=.3,
                                           noise=.05)
            data = datasets.base.Bunch(data=xx, target=yy)
        elif opt.lower() == 'moons':
            if n_samples == 0: n_samples = 400
            xx, yy = datasets.make_moons(n_samples=n_samples, noise=.01)
            data = datasets.base.Bunch(data=xx, target=yy)
        elif opt.lower() == 'custom':
            data = load_custom(x_filename, y_filename)
    except IOError as e:
        print("I/O error({0}): {1}".format(e.errno, e.strerror))

    X, y = data.data, data.target
    if y is not None and n_samples > 0 and X.shape[0] > n_samples:
        try:  # Legacy for sklearn
            sss = StratifiedShuffleSplit(y, test_size=n_samples, n_iter=1)
            # idx = np.random.permutation(X.shape[0])[:n_samples]
        except TypeError:
            sss = StratifiedShuffleSplit(n_iter=1, test_size=n_samples) \
                  .split(X, y)

        _, idx = list(sss)[0]
        X, y = X[idx, :], y[idx]

    try:
        feat_names = data.features_names
    except:
        feat_names = range(0, X.shape[1])
    try:
        class_names = data.target_names
    except:
        class_names = 0

    return X, y, feat_names, class_names
