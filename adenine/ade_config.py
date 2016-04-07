#!/usr/bin/python
# -*- coding: utf-8 -*-

from adenine.utils import data_source
from adenine.utils import extra

# --------------------------  EXPERMIENT INFO ------------------------- #
exp_tag = 'cool_experiment'
output_root_folder = 'results'

# ----------------------------  INPUT DATA ---------------------------- #
data_file = 'X.csv'
labels_file = 'y.csv' # OPTIONAL
X, y, feat_names, class_names = data_source.load('custom', data_file, labels_file)

# -----------------------  PIPELINE DEFINITION ------------------------ #

# --- Missing Values Imputing --- #
step0 = {'Impute': [False], 'Missing': [-1], 'Replacement': ['median','mean']}

# --- Data Preprocessing --- #
step1 = {'None': [False], 'Recenter': [False], 'Standardize': [False],
         'Normalize': [True, ['l2']], 'MinMax': [False, [0,1]]}

# --- Dimensionality Reduction & Manifold Learning --- #
step2 = {'PCA': [False], 'IncrementalPCA': [False], 'RandomizedPCA': [False],
         'KernelPCA': [True, ['linear','rbf','poly']], 'Isomap': [False],
         'LLE': [False, ['standard','modified','hessian', 'ltsa']],
         'SE': [False], 'MDS': [False, ['metric','nonmetric']],
         'tSNE': [False], 'None': [False]}

# --- Clustering --- #
step3 = {'KMeans': [True, [5]],
         'AP': [False]
        #  'AP': [False, [1, ['precomputed']]],
         'MS': [False],
         'Spectral': [True, [5]]
        #  'Spectral': [True, [5, ['precomputed']]],
         'Hierarchical': [False, [3, ['manhattan','euclidean'], ['ward','complete','average']]]}
        #  'Hierarchical': [False, [3, ['precomputed'], ['complete','average']]]}
