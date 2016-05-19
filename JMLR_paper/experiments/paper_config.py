#!/usr/bin/env python
# -*- coding: utf-8 -*-

from adenine.utils import data_source
from adenine.utils import extra

# --------------------------  EXPERMIENT INFO ------------------------- #
exp_tag = 'demo'
output_root_folder = 'exp_figures'
file_format = 'pdf'
plotting_context = 'paper'

# ----------------------------  INPUT DATA ---------------------------- #
X, y, feat_names, class_names = data_source.load('circles')

# -----------------------  PIPELINE DEFINITION ------------------------ #

# # --- Missing Values Imputing --- #
step0 = {'Impute': [False, {}]}

# --- Data Preprocessing --- #
step1 = {'None': [True]}

# --- Dimensionality Reduction & Manifold Learning --- #
step2 = {'KernelPCA': [True, {'n_components': 2,
                              'kernel': ['linear','rbf'],
                              'gamma': 2}]
        }

# --- Clustering --- #
step3 = {'KMeans': [True, {'n_clusters': [2]}]}
