#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from adenine.utils import data_source
from adenine.utils.extensions import Imputer
# from sklearn.preprocessing import Imputer

def main():
    """
    Testing the KNN data imputing.
    """
    Xreal, y, feat_names, class_names = data_source.load('gauss')
    [n,p] = Xreal.shape
    print("{} x {} matrix loaded".format(n,p))


    # Choose the missing rate
    missing_rate = 0.2
    n_missing = int(missing_rate * (n*p))

    # Create holes in the matrix
    idx = np.random.permutation(n*p);
    xx = Xreal.ravel().copy()
    xx[idx[:n_missing]] = np.nan
    X = np.reshape(xx, (n,p))
    print("{} values deleted".format(n_missing))

    # Start test
    strategies = ["mean", "median", "most_frequent", "nearest_neighbors"]

    imp = Imputer(strategy=strategies[3])
    Ximp = imp.fit_transform(X)

    if len(np.where(np.isnan(Ximp))[0]) == 0:
        print("All values were imputed according to: {}-strategy".format(imp.strategy))
    else:
        print("Empty values: {}".format(len(np.where(np.isnan(Ximp))[0])))

    # Check results
    dist = np.sqrt(np.sum((Xreal.ravel() - Ximp.ravel())**2))
    print("dist(Xreal - Ximp) = {}".format(dist))

    # print "---------------------------------"
    # print X
    # print "---------------------------------"
    # print Ximp


if __name__ == '__main__':
    main()
