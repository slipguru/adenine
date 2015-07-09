#!/usr/bin/python
# -*- coding: utf-8 -*-

# This module is just a wrapper for some sklearn.datasets functions

import numpy as np
from sklearn import datasets

class dataSetObj:
    """Sklearn-like object.
    
    A simple dictionary-like object that contains a custom dataset. The meaningful attributes are .data and .target.
    
    Parameters
    -----------
    D : array of float, shape : n_samples x n_features
        The input data matrix.
        
    T :  array of float, shape : n_samples
        The label vector.
    """
    def __init__(self,D,T):
        self.data = D
        self.target = T
        
def load_custom(fileName_X = 'X.npy', fileName_y = 'y.npy'):
    """Load a custom dataset.
    
    This function loads the data matrix and the label vector returning a unique sklearn-like object dataSetObj.
    
    Parameters
    -----------
    fileName_X : string, default : 'X.npy'
        The data matrix file name.
    
    fileName_y : string, default : 'y.npy'
        The label vector file name.
        
    Returns
    -----------
    data : dataSetObj
        An instance of the dataSetObj class, the meaningful attributes are .data, the data matrix, and .target, the label vector
    """
    try: # labels are not mandatory
        y = np.load(fileName_y)
    except IOError as e:
        y = np.nan
        e.strerror = "No labels file provided"
        print("I/O error({0}): {1}".format(e.errno, e.strerror))
        
    return dataSetObj(np.load(fileName_X),y)
    

def load(opt = 'custom', fileName_X = 'X.npy', fileName_y = 'y.npy'):
    """Load a specified dataset.
    
    This function can be used either to load one of the standard scikit-learn datasets or a different dataset saved as X.npy Y.npy in the working directory.
    
    Parameters
    -----------
    opt : {'iris', 'digits', 'diabetes', 'boston', 'blobs','custom'}, default: 'custom'
    
    fileName_X : string, default : 'X.npy'
        The data matrix file name.
    
    fileName_y : string, default : 'y.npy'
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
    try: # Select the dataset
        if opt.lower() == 'iris':
            data = datasets.load_iris()
        elif opt.lower() == 'digits':
            data = datasets.load_digits()
        elif opt.lower() == 'diabetes':
            data = datasets.load_diabetes()
        elif opt.lower() == 'boston':
            data = datasets.load_boston()
        elif opt.lower() == 'blobs':
            xx, yy = datasets.make_blobs(n_samples=500, centers=[[1, 1], [-1, -1], [1, -1]], cluster_std=0.3, n_features=3)
            # xx, yy = datasets.make_classification(n_samples = 500, n_features = 20, n_informative = 2)
            data = datasets.base.Bunch(data = xx, target = yy)
        elif opt.lower() == 'custom':
            data = load_custom(fileName_X, fileName_y)
    except IOError as e:
         print("I/O error({0}): {1}".format(e.errno, e.strerror))
    
    # Get X, y and feature_names
    X, y = data.data, data.target
    try:
        feat_names = data.features_names
        class_names = data.target_names
    except:
        feat_names = range(0,X.shape[1])
        class_names = 0
        
    return X, y, feat_names, class_names
