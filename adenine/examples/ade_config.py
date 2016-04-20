#!/usr/bin/python
# -*- coding: utf-8 -*-

from adenine.utils import data_source
from adenine.utils import extra

# --------------------------  EXPERMIENT INFO ------------------------- #
exp_tag = 'debug'
output_root_folder = 'results'

# ----------------------------  INPUT DATA ---------------------------- #
# X, y, feat_names, class_names = data_source.load('iris', n_samples=100)
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
         'Normalize': [True, ['l2']], 'MinMax': [False, [0,1]]}

# --- Dimensionality Reduction & Manifold Learning --- #
step2 = {'PCA': [False], 'IncrementalPCA': [False], 'RandomizedPCA': [False],
         'KernelPCA': [True, ['linear','rbf','poly']], 'Isomap': [False],
        #  'LLE': [True, ['standard','modified','hessian', 'ltsa']],
         'LLE': [False, ['standard','modified']],
         'SE': [False], 'MDS': [False, ['metric','nonmetric']],
         'tSNE': [False], 'None': [False]}

# --- Clustering --- #
step3 = {'KMeans': [True, ['auto']], # cannot be 'precomputed'
        #  'AP': [False, [1,'precomputed']], # can be 'precomputed'
         'AP': [False, ['auto']], # can be 'precomputed'
         'MS': [False], # cannot be 'precomputed'
        #  'Spectral': [True, [50, ['precomputed']]], # can be 'precomputed'
         'Spectral': [False, [3]], # can be 'precomputed'
         'Hierarchical': [False, [3, ['manhattan','euclidean'], ['ward','complete','average']]]}
        #  'Hierarchical': [False, [3, ['precomputed']]] # can be 'precomputed'
