#!/usr/bin/python
# -*- coding: utf-8 -*-

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Federico Tomasi, Annalisa Barla
#
# FreeBSD License
######################################################################

from __future__ import division

import numpy as np

from adenine.utils import data_source
from adenine.utils.extensions import Imputer


def test(missing_rate):
    """
    Testing the KNN data imputing.
    """
    Xreal, y, feat_names, class_names = data_source.load('iris')
    # Xreal, y, feat_names, class_names = data_source.load('gauss', n_samples=100)
    n, p = Xreal.shape
    print("{} x {} matrix loaded".format(n, p))

    # Choose the missing rate
    # missing_rate = 0.5
    n_missing = int(missing_rate * (n*p))

    # Create holes in the matrix
    np.random.seed(42)
    idx = np.random.permutation(n*p)
    xx = Xreal.ravel().copy()
    xx[idx[:n_missing]] = np.nan
    X = np.reshape(xx, (n, p))
    print("{} values deleted".format(n_missing))

    # Save data
    np.savetxt('X_missing.csv', X, delimiter=',')
    np.savetxt('Y_missing_test.csv', y, delimiter=',')

    # Start test
    strategies = ["mean", "median", "most_frequent", "nearest_neighbors"]

    imp = Imputer(strategy=strategies[3])
    Ximp = imp.fit_transform(X)

    if len(np.where(np.isnan(Ximp))[0]) == 0:
        print("All values were imputed according to: {}-strategy".format(imp.strategy))
    else:
        print("Empty values: {}".format(len(np.where(np.isnan(Ximp))[0])))

    # Check results
    dist = np.sqrt(np.sum((Xreal[imp.X_mask,:].ravel() - Ximp.ravel())**2))
    print("dist(Xreal - Ximp) = {}".format(dist))

    # print(Ximp)


def main():
    for missing_rate in np.linspace(0.01, 0.3, 2):
        print("\nmissing rate: {}".format(missing_rate))
        test(missing_rate)


if __name__ == '__main__':
    main()
