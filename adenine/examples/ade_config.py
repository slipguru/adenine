#!/usr/bin/env python
# -*- coding: utf-8 -*-

from adenine.utils import data_source
from adenine.utils import extra

# --------------------------  EXPERMIENT INFO ------------------------- #
exp_tag = 'debug'
output_root_folder = 'results'
file_format = 'pdf' # or 'png'
plotting_context = 'paper' # one of {paper, notebook, talk, poster}

# ----------------------------  INPUT DATA ---------------------------- #
#X, y, feat_names, class_names = data_source.load('iris')
#X, y, feat_names, class_names = data_source.load('gauss', n_samples=300)
# X, y, feat_names, class_names = data_source.load('circles')
X, y, feat_names, class_names = data_source.load('digits')
# X, y, feat_names, class_names = data_source.load('diabetes')
# X, y, feat_names, class_names = data_source.load('boston')
# X, y, feat_names, class_names = data_source.load('custom', 'data/X.npy', 'data/y.npy')
# X, y, feat_names, class_names = data_source.load('custom', 'data/X.csv', 'data/y.csv')

# X, y, feat_names, class_names = data_source.load('custom', '/home/fede/src/adenine/adenine/examples/TM_matrix.csv')
# X = extra.ensure_symmetry(X)
# X = 1. - X  # i want affinity

# -----------------------  PIPELINE DEFINITION ------------------------ #

# --- Missing Values Imputing --- #
# step0 = {'Impute': [False, {'missing_values': 'NaN',
                            # 'strategy': ['median','mean','nearest_neighbors']}]}

# --- Data Preprocessing --- #
step1 = {'None': [False], 'Recenter': [False], 'Standardize': [False],
         'Normalize': [True, {'norm': ['l2']}],
         'MinMax': [False, {'feature_range': [(0,1), (-1,1)]}]}

# --- Dimensionality Reduction & Manifold Learning --- #
step2 = {'PCA': [False, {'n_components': 3}],
         'IncrementalPCA': [False, {'n_components': 3}],
         'RandomizedPCA':  [False, {'n_components': 3}],
         'KernelPCA':      [True, {'n_components': 2,
                                    'kernel': ['linear','rbf','poly'], 'gamma': 2}],
         'Isomap': [False, {'n_components': 3, 'n_neighbors': 5}],
         'LLE':    [False, {'n_components': 3, 'n_neighbors': 5, # xxx
                            'method': ['standard','modified','hessian','ltsa']}],
         'SE':   [False, {'n_components': 3, 'affinity': ['nearest_neighbors','rbf']}], # can be 'precomputed'
         'MDS':  [False, {'n_components': 3, 'metric': [True, False]}],
         'tSNE': [False, {'n_components': 3}],
         'None': [False, {}]
         }

# --- Clustering --- #
step3 = {'KMeans': [True, {'n_clusters': ['auto']}], # cannot be 'precomputed'
         'AP': [False, {'preference': ['auto']}], # can be 'precomputed'
         'MS': [False], # cannot be 'precomputed'
         'Spectral': [False, {'n_clusters': [2]}], # can be 'precomputed'
         'Hierarchical': [False, {'n_clusters': [10],
                                  #'affinity': ['manhattan','euclidean'],
                                  'affinity': ['euclidean'],
                                  #'linkage':  ['ward','complete','average']}]
                                  'linkage':  ['ward','average']}]
        }
