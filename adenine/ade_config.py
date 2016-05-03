#!/usr/bin/env python
# -*- coding: utf-8 -*-

from adenine.utils import data_source
from adenine.utils import extra

from numpy import nan
# --------------------------  EXPERMIENT INFO ------------------------- #
exp_tag = 'cool_experiment'
output_root_folder = 'results'
plotting_context = 'notebook' # one of {paper, notebook, talk, poster}

# ----------------------------  INPUT DATA ---------------------------- #
data_file = 'X.csv'
labels_file = 'y.csv' # OPTIONAL
X, y, feat_names, class_names = data_source.load('custom', data_file, labels_file)

# -----------------------  PIPELINE DEFINITION ------------------------ #

# --- Missing Values Imputing --- #
step0 = {'Impute': [False], 'Missing': [nan], 'Replacement': ['median','mean','nearest_neighbors']}

# --- Data Preprocessing --- #
step1 = {'None': [False], 'Recenter': [False], 'Standardize': [False],
         'Normalize': [False, ['l2']], 'MinMax': [False, [0,1]]}

# --- Dimensionality Reduction & Manifold Learning --- #
step2 = {'PCA': [False, {'n_components': 3}],
         'IncrementalPCA': [False],
         'RandomizedPCA': [False],
         'KernelPCA': [False, {'kernel': ['linear','rbf','poly']}],
         'Isomap': [False, {'n_neighbors': 5}],
         'LLE': [False, {'n_neighbors': 5, 'method': ['standard','modified','hessian','ltsa']}],
         'SE': [False, {'affinity': ['nearest_neighbors','rbf']}],  # affinity can be 'precomputed'
         'MDS': [False, {'metric': True}],
         'tSNE': [False],
         'None': [False]
         }

# --- Clustering --- #
step3 = {'KMeans': [False, {'n_clusters': [3, 'auto']}],
         'AP': [False, {'preference': ['auto']}],                        # affinity can be 'precomputed'
         'MS': [False],                                                  # affinity can be 'precomputed'
         'Spectral': [False, {'n_clusters': [3, 8]}],                    # affinity can be 'precomputed'
         'Hierarchical': [False, {'n_clusters': [3, 8],
                                  'affinity': ['manhattan','euclidean'], # affinity can be 'precomputed'
                                  'linkage':  ['ward','complete','average']}]
        }
