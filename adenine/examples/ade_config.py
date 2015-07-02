#!/usr/bin/python
# -*- coding: utf-8 -*-

from adenine.utils import data_source

# --------------------------  EXPERMIENT INFO ------------------------- #
exp_tag = 'debug'
output_root_folder = 'results'

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
         'KernelPCA': [True, ['linear','rbf','poly']], 'Isomap': [True],
         'LLE': [True, ['standard','modified','hessian', 'ltsa']],
         'SE': [True], 'MDS': [True, ['metric','nonmetric']],
         'tSNE': [True], 'None': [True]}

# --- Clustering --- #
step3 = {'KMeans': [True, [3]],
         'KernelKMeans': [True, [3,['rbf','poly']]], #TODO
         'AP': [True], 'MS': [True], 'Spectral': [True, [3]],
         'Hierarchical': [True, [3, ['manhattan','euclidean'],
                                 ['ward','complete','average']]]}
