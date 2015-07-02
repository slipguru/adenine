#!/usr/bin/python
# -*- coding: utf-8 -*-

from adenine.utils import data_source

# --------------------------  EXPERMIENT INFO ------------------------- #
exp_tag = 'debug'
output_root_folder = 'results'
# parallel = False

# ----------------------------  INPUT DATA ---------------------------- #
X, y, feat_names = data_source.load('iris')
# X, y, feat_names = data_source.load('digits')
# X, y, feat_names = data_source.load('diabetes')
# X, y, feat_names = data_source.load('boston')
# X, y, feat_names = data_source.load('custom', 'bigX.npy', 'bigY.npy')

# -----------------------  PIPELINE DEFINITION ------------------------ #

# --- Missing Values Imputing --- #
step0 = {'Impute': [False], 'Missing': [-1], 'Replacement': ['median','mean']}

# --- Data Preprocessing --- #
step1 = {'None': [False], 'Recenter': [False], 'Standardize': [False],
         'Normalize': [False, ['l2']], 'MinMax': [True, [0,1]]}

# --- Dimensionality Reduction & Manifold Learning --- #
step2 = {'PCA': [False], 'IncrementalPCA': [False], 'RandomizedPCA': [False],
         'KernelPCA': [True, ['linear','rbf','poly']], 'Isomap': [False],
         'LLE': [False, ['standard','modified','hessian', 'ltsa']],
         'SE': [False], 'MDS': [False, ['metric','nonmetric']],
         'tSNE': [False], 'None': [False]}

# --- Clustering --- #
step3 = {'KMeans': [False, [3]],
         'KernelKMeans': [False, [3,['rbf','poly']]], #TODO
         'AP': [False], 'MS': [False], 'Spectral': [False, [3]],
         'Hierarchical': [True, [3, ['manhattan','euclidean'],
                                 ['ward','complete','average']]]}
