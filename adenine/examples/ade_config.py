#!/usr/bin/python
# -*- coding: utf-8 -*-

from adenine.utils import data_source

# --------------------------  EXPERMIENT INFO ------------------------- #
exp_tag = 'debug_csv'
output_root_folder = 'results'
# parallel = False

# ----------------------------  INPUT DATA ---------------------------- #
# X, y, feat_names, class_names = data_source.load('iris')
# X, y, feat_names, class_names = data_source.load('gauss')
# X, y, feat_names, class_names = data_source.load('digits')
# X, y, feat_names, class_names = data_source.load('diabetes')
# X, y, feat_names, class_names = data_source.load('boston')
# X, y, feat_names, class_names = data_source.load('custom', 'X.npy', 'y.npy')
# X, y, feat_names, class_names = data_source.load('custom', 'X.csv', 'y.csv')
X, y, feat_names, class_names = data_source.load('custom', '/home/fede/src/adenine/adenine/examples/TM_matrix.csv')
if not (X.T == X).all():
    X = (X.T + X) / 2.
    X = 1. - X

# -----------------------  PIPELINE DEFINITION ------------------------ #

# --- Missing Values Imputing --- #
step0 = {'Impute': [False], 'Missing': [-1], 'Replacement': ['median','mean']}

# --- Data Preprocessing --- #
step1 = {'None': [True], 'Recenter': [False], 'Standardize': [False],
         'Normalize': [False, ['l2']], 'MinMax': [False, [0,1]]}

# --- Dimensionality Reduction & Manifold Learning --- #
step2 = {'PCA': [False], 'IncrementalPCA': [False], 'RandomizedPCA': [False],
         'KernelPCA': [False, ['linear','rbf','poly']], 'Isomap': [False],
         'LLE': [False, ['standard','modified','hessian', 'ltsa']],
         'SE': [False], 'MDS': [False, ['metric','nonmetric']],
         'tSNE': [False], 'None': [True]}

# --- Clustering --- #
step3 = {'KMeans': [False, [5]],
         'KernelKMeans': [False, [3,['rbf','poly']]], #TODO
         'AP': [True, ['precomputed']], 'MS': [False],
         'Spectral': [True, [3, ['precomputed']]],
         #'Hierarchical': [False, [3, ['manhattan','euclidean'], ['ward','complete','average']]]
         'Hierarchical': [True, [3, ['precomputed']]]
         }
