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