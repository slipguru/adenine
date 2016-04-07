#!/usr/bin/python
# -*- coding: utf-8 -*-

from adenine.utils import data_source
from adenine.utils import extra

# --------------------------  EXPERMIENT INFO ------------------------- #
exp_tag = 'debug_csv'
output_root_folder = 'results'

# ----------------------------  INPUT DATA ---------------------------- #
# X, y, feat_names, class_names = data_source.load('iris')
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
step0 = {'Impute': [False], 'Missing': [-1], 'Replacement': ['median','mean']}

# --- Data Preprocessing --- #
step1 = {'None': [True], 'Recenter': [False], 'Standardize': [False],
         'Normalize': [False, ['l2']], 'MinMax': [False, [0,1]]}

# --- Dimensionality Reduction & Manifold Learning --- #
step2 = {'PCA': [True], 'IncrementalPCA': [False], 'RandomizedPCA': [False],
         'KernelPCA': [True, ['linear','rbf','poly']], 'Isomap': [False],
         'LLE': [False, ['standard','modified','hessian', 'ltsa']],
         'SE': [False], 'MDS': [False, ['metric','nonmetric']],
         'tSNE': [False], 'None': [True]}

# --- Clustering --- #
step3 = {'KMeans': [True, [3]], # cannot be 'precomputed'
        #  'AP': [False, [1,'precomputed']], # can be 'precomputed'
         'AP': [False], # can be 'precomputed'
         'MS': [False], # cannot be 'precomputed'
        #  'Spectral': [True, [50, ['precomputed']]], # can be 'precomputed'
         'Spectral': [True, [3]], # can be 'precomputed'
         'Hierarchical': [False, [3, ['manhattan','euclidean'], ['ward','complete','average']]]}
        #  'Hierarchical': [False, [3, ['precomputed']]] # can be 'precomputed'
