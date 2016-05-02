#!/usr/bin/env python
# -*- coding: utf-8 -*-

from adenine.utils import data_source
from adenine.utils import extra

# --------------------------  EXPERMIENT INFO ------------------------- #
exp_tag = 'debug'
output_root_folder = 'results'

# ----------------------------  INPUT DATA ---------------------------- #
# X, y, feat_names, class_names = data_source.load('iris' , n_samples=100)
X, y, feat_names, class_names = data_source.load('gauss')
# X, y, feat_names, class_names = data_source.load('digits')
# X, y, feat_names, class_names = data_source.load('diabetes')
# X, y, feat_names, class_names = data_source.load('boston')
# X, y, feat_names, class_names = data_source.load('custom', 'X.npy', 'y.npy')
# X, y, feat_names, class_names = data_source.load('custom', 'X.csv', 'y.csv')

# X, y, feat_names, class_names = data_source.load('custom', '/home/fede/src/adenine/adenine/examples/TM_matrix.csv')
# X = extra.ensure_symmetry(X)
# X = 1. - X  # i want affinity

# -----------------------  PIPELINE DEFINITION ------------------------ #

# --- Missing Values Imputing --- #
step0 = {'Impute': [False], 'Missing': [-1], 'Replacement': ['median','mean','nearest_neighbors']}

# --- Data Preprocessing --- #
step1 = {'None': [False], 'Recenter': [False], 'Standardize': [False],
         'Normalize': [False, ['l2']], 'MinMax': [True, [0,1]]}

# --- Dimensionality Reduction & Manifold Learning --- #
step2 = {'PCA': [True, {'n_components': 3}],
         'IncrementalPCA': [False, {'n_components': 3}],
         'RandomizedPCA': [False, {'n_components': 3}],
         'KernelPCA': [False, {'n_components': 3,
                              'kernel': ['linear','rbf','poly']}],
         'Isomap': [False, {'n_components': 3, 'n_neighbors': 5}],
         'LLE': [False, {'n_components': 3, 'n_neighbors': 5, # xxx
                         'method': ['standard','modified','hessian','ltsa']}],
         'SE': [False, {'n_components': 3, 'affinity': ['nearest_neighbors','rbf']}], # can be 'precomputed'
         'MDS': [False, {'n_components': 3, 'metric': [True, False]}],
         'tSNE': [False, {'n_components': 3}],
         'None': [True, {}]
         }

# --- Clustering --- #
step3 = {'KMeans': [True, {'n_clusters': ['auto', 3]}], # cannot be 'precomputed'
        #  'AP': [False, [1,'precomputed']], # can be 'precomputed'
         'AP': [True, {'preference': ['auto', -40]}], # can be 'precomputed'
         'MS': [False], # cannot be 'precomputed'
        #  'Spectral': [True, [50, ['precomputed']]], # can be 'precomputed'
         'Spectral': [False, {'n_clusters': [3, 8]}], # can be 'precomputed'
        #  'Hierarchical': [False, [3, ['manhattan','euclidean'], ['ward','complete','average']]]}
         'Hierarchical': [False, {'n_clusters': [3, 8],
                                  'affinity': ['manhattan','euclidean'],
                                  'linkage':  ['ward','complete','average']}]
        }
