#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

from adenine.utils import data_source
from adenine.utils.extensions import Imputer
# from sklearn.preprocessing import Imputer

def main():
    """
    Testing the KNN data imputing.
    """
    X, y, feat_names, class_names = data_source.load('gauss')
    [n,p] = X.shape
    print("{} x {} matrix created".format(n,p))

    # Choose the missing rate
    missing_rate = 0.01
    n_missing = int(missing_rate * (n*p))

    # Create holes in the matrix
    idx = np.random.permutation(n*p);
    xx = X.ravel()
    xx[idx[:n_missing]] = np.nan
    X = np.reshape(xx, (n,p))
    # X[0,0] = np.nan
    # X[0,1] = np.nan
    print("{} values deleted".format(n_missing))

    # Start test
    strategies = ["mean", "median", "most_frequent", "nearest_neighbors"]

    imp = Imputer(strategy=strategies[3])

    X = imp.fit_transform(X)
    # X = imp.fit(X).transform(X)

    if len(np.where(np.isnan(X))[0]) == 0:
        print("All values were imputed according to: {}-strategy".format(imp.strategy))
    else:
        print("Empty values: {}".format(len(np.where(np.isnan(X))[0])))








if __name__ == '__main__':
    main()
