#!/usr/bin/python
# -*- coding: utf-8 -*-

# This module is just a wrapper for some sklearn.datasets functions

import numpy as np
from sklearn import datasets

def load(opt):

    if opt.lower() == 'gauss':
        X = 1
        y = 2
    elif opt.lower() == 'iris':
        data = datasets.load_iris()
        X = data.X
        y = data.target
    else:
        # Look for X.npy and y.npy in the current directory
        X = np.array([])
        y = np.array([])

    return X, y, feat_names
