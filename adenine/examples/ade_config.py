#!/usr/bin/python
# -*- coding: utf-8 -*-

from adenine.utils import data_source

# --- tmp stuff --- #
reload(data_source)
# --- tmp stuff --- #

# ----------------------------  INPUT DATA ---------------------------- #
X, y, feat_names = data_source.load('iris')
# X, y, feat_names = data_source.load('digits')
# X, y, feat_names = data_source.load('diabetes')
# X, y, feat_names = data_source.load('boston')
# X, y, feat_names = data_source.load('custom')

# -----------------------  PIPELINE DEFINITION ------------------------ #

# --- Missing Values Imputing --- #
step0 = {'Impute': [False], 'Missing': [-1], 'Replacement': ['median','mean']}

# --- Data Preprocessing --- #
step1 = {'None': [True], 'Recenter': [False], 'Standardize': [False],
         'Normalize': [False, ['l2']], 'MinMax': [True, [0,1]]}

# --- Dimensionality Reduction & Manifold Learning --- #
step2 = {'PCA': [False], 'IncrementalPCA': [False], 'RandomizedPCA': [False],
         'KernelPCA': [True, ['rbf','poly']], 'Isomap': [False],
         'LLE': [False, ['standard','modified','hessian']],
         'SE': [False], 'LTSA': [False],
         'MDS': [False, ['metric','nonmetric']],
         'tSNE': [False], 'None': [False]}

# --- Clustering --- #
step3 = {'KMeans': [False, [3]],
         'KernelKMeans': [False, ['rbf','poly']],
         'AP': [False], 'MS': [False], 'Spectral': [False],
         'Hierarchical': [False, ['ward','complete','average'],
                                 ['manhattan','euclidean','minkowski']]}
