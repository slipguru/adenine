#!/usr/bin/env python
# -*- coding: utf-8 -*-

from adenine.utils import data_source
from adenine.utils import extra

# --------------------------  EXPERMIENT INFO ------------------------- #
exp_tag = 'debug'
output_root_folder = 'results'
<<<<<<< Updated upstream
plotting_context = 'poster' # one of {paper, notebook, talk, poster}
file_format = 'pdf' # or 'png'
=======
plotting_context = 'paper' # one of {paper, notebook, talk, poster}
>>>>>>> Stashed changes

# ----------------------------  INPUT DATA ---------------------------- #
# X, y, feat_names, class_names = data_source.load('iris' , n_samples=100)
# X, y, feat_names, class_names = data_source.load('gauss')
X, y, feat_names, class_names = data_source.load('circles')
# X, y, feat_names, class_names = data_source.load('digits')
# X, y, feat_names, class_names = data_source.load('diabetes')
# X, y, feat_names, class_names = data_source.load('boston')
# X, y, feat_names, class_names = data_source.load('custom', 'data/X.npy', 'data/y.npy')
# X, y, feat_names, class_names = data_source.load('custom', 'data/X.csv', 'data/y.csv')

# X, y, feat_names, class_names = data_source.load('custom', '/home/fede/src/adenine/adenine/examples/TM_matrix.csv')
# X = extra.ensure_symmetry(X)
# X = 1. - X  # i want affinity

# -----------------------  PIPELINE DEFINITION ------------------------ #

# --- Missing Values Imputing --- #
step0 = {'Impute': [True, {'missing_values': 'NaN',
                            'strategy': ['median','mean','nearest_neighbors']}]}

# --- Data Preprocessing --- #
<<<<<<< Updated upstream
step1 = {'None': [False], 'Recenter': [False], 'Standardize': [False],
         'Normalize': [False, {'norm': ['l1','l2']}],
         'MinMax': [True, {'feature_range': [(0,1), (-1,1)]}]}
=======
step1 = {'None': [False], 'Recenter': [True], 'Standardize': [False],
         'Normalize': [False, ['l2']], 'MinMax': [False, [0,1]]}
>>>>>>> Stashed changes

# --- Dimensionality Reduction & Manifold Learning --- #
step2 = {'PCA': [False, {'n_components': 3}],
         'IncrementalPCA': [False, {'n_components': 3}],
         'RandomizedPCA':  [False, {'n_components': 3}],
<<<<<<< Updated upstream
<<<<<<< HEAD
         'KernelPCA':      [False, {'n_components': 7,
=======
         'KernelPCA':      [True, {'n_components': 3,
>>>>>>> JMLR_paper_preparation
=======
         'KernelPCA':      [True, {'n_components': 2,
>>>>>>> Stashed changes
                                    'kernel': ['linear','rbf','poly']}],
         'Isomap': [False, {'n_components': 3, 'n_neighbors': 5}],
         'LLE':    [False, {'n_components': 3, 'n_neighbors': 5, # xxx
                            'method': ['standard','modified','hessian','ltsa']}],
         'SE':   [False, {'n_components': 3, 'affinity': ['nearest_neighbors','rbf']}], # can be 'precomputed'
         'MDS':  [False, {'n_components': 3, 'metric': [True, False]}],
         'tSNE': [False, {'n_components': 3}],
         'None': [False, {}]
         }

# --- Clustering --- #
<<<<<<< Updated upstream
<<<<<<< HEAD
step3 = {'KMeans': [True, {'n_clusters': ['auto', 5]}], # cannot be 'precomputed'
=======
step3 = {'KMeans': [False, {'n_clusters': ['auto', 3]}], # cannot be 'precomputed'
>>>>>>> JMLR_paper_preparation
=======
step3 = {'KMeans': [True, {'n_clusters': ['auto', 2]}], # cannot be 'precomputed'
>>>>>>> Stashed changes
        #  'AP': [False, [1,'precomputed']], # can be 'precomputed'
         'AP': [False, {'preference': ['auto']}], # can be 'precomputed'
         'MS': [False], # cannot be 'precomputed'
        #  'Spectral': [True, [50, ['precomputed']]], # can be 'precomputed'
<<<<<<< Updated upstream
         'Spectral': [False, {'n_clusters': [10]}], # can be 'precomputed'
=======
         'Spectral': [True, {'n_clusters': [2]}], # can be 'precomputed'
>>>>>>> Stashed changes
        #  'Hierarchical': [False, [3, ['manhattan','euclidean'], ['ward','complete','average']]]}
         'Hierarchical': [True, {'n_clusters': [2],
                                  'affinity': ['manhattan','euclidean'],
                                  'linkage':  ['ward','complete','average']}]
        }
