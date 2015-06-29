#!/usr/bin/python
# -*- coding: utf-8 -*-

from adenine.utils import toy_data

# ----------------------------  INPUT DATA ---------------------------- #
X, y, feat_names = toy_data.load('iris')
X, y, feat_names = toy_data.load('digits')
X, y, feat_names = toy_data.load('diabetes')
X, y, feat_names = toy_data.load('boston')
X, y, feat_names = toy_data.load('custom')