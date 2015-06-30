#!/usr/bin/python
# -*- coding: utf-8 -*-

from adenine.utils import data_source

# ----------------------------  INPUT DATA ---------------------------- #
X, y, feat_names = data_source.load('iris')
# X, y, feat_names = data_source.load('digits')
# X, y, feat_names = data_source.load('diabetes')
# X, y, feat_names = data_source.load('boston')
# X, y, feat_names = data_source.load('custom')

# -----------------------  PIPELINE DEFINITION ------------------------ #

# --- Missing Values Imputing --- #
step0 = {'Impute': [True], 'Missing': [-1], 'Replacement': ['median','mean']}

# --- Data Preprocessing --- #
step1 = {'None': [True], 'Recenter': [True], 'Standardize': [True],
         'Normalize': [True, ['l2']], 'MinMax': [True, [0,1]]}

# --- Dimensionality Reduction & Manifold Learning --- #
step2 = {'PCA': [True], 'IncrementalPCA': [True], 'RandomizedPCA': [True],
         'KernelPCA': [True, ['rbf','poly']], 'Isomap': [True],
         'LLE': [True, ['standard','modified','hessian']],
         'SE': [True], 'LTSA': [True],
         'MDS': [True, ['metric','nonmetric']],
         'tSNE': [True], 'None': [True]}

# --- Clustering --- #
step3 = {'KMeans': [False, [3]],
         'KernelKMeans': [False, ['rbf','poly']],
         'AP': [False], 'MS': [False], 'Spectral': [False],
         'Hierarchical': [False, ['ward','complete','average'],
                                 ['manhattan','euclidean','minkowski']]}
