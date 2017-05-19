#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module contains utility functions for GEO DataSets wrangling."""

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Federico Tomasi, Annalisa Barla
#
# FreeBSD License
######################################################################

import GEOparse
import logging
import os
import pandas as pd
from sklearn import datasets


def get_GEO(accession_number):
    """Get the GEO data from its accession number.

    Parameters
    -----------
    accession_number : string
        'GSEXXXXX' is any GEO accession ID loaded by `GEOparse`.
    """
    gse = GEOparse.get_GEO(geo=accession_number, destdir=os.curdir,
                           silent=True, include_data=True,
                           how='full')
    xx = gse.pivot_samples('VALUE').transpose()
    index = xx.index.tolist()
    feature_names = xx.columns.tolist()
    yy = gse.phenotype_data['title']
    data = datasets.base.Bunch(data=xx.values, target=yy.values,
                               feature_names=feature_names,
                               index=index)
    return data


def label_mapper(raw_labels, new_labels):
    """Map some raw labels into new labels.

    When dealing with GEO DataSets it is very common that each GSM sample has
    a different phenotye (e.g. 'Brain - 001', 'Brain - 002', ...). This
    function maps these raw labels into new homogeneous labels.

    Parameters
    -----------
    raw_labels : list of strings
        list of unpreprocessed labels
    new_labels : list of strings
        list of labels to map

    Returns
    -----------
    y : array of float, shape : n_samples
        the modified label vector

    Examples
    -----------
    >>> raw_labels = ['Brain - 001', 'Brain - 002', 'Muscle - 001', 'Muscle - 002']
    >>> label_mapper(raw_labels, ['Brain', 'Muscle'])
    ['Brain', 'Brain', 'Muscle', 'Muscle']
    """
    y = []
    for rl in raw_labels:
        for nl in new_labels:
            if nl in rl:
                y.append(nl)
                break
        else:
            y.append(rl)
            # print('No mapping rule for %s', rl)
    return y


def GEO_select_samples(data, labels, selected_labels, index,
                       feature_names=None):
    """GEO DataSets data selection tool.

    Modify the labels with `label_mapper` then return only the samples with
    labels in selected_labels.

    Parameters
    -----------
    data : array of float, shape : n_samples x n_features
        the dataset
    labels : numpy array (n_samples,)
        the labels vector
    selected_labels : list of strings
        a subset of new_labels containing only the samples wanted in the
        final dataset
    index : list of strings
        the sample indexes
    feature_names : list of strings
        the feature set
    samples_on : string in ['col', 'cols', 'row', 'rows']
        wether the samples are on columns or rows

    Returns
    -----------
    data : sklearn.datasets.base.Bunch
        An instance of the sklearn.datasets.base.Bunch class, the meaningful
        attributes are .data, the data matrix, and .target, the label vector.
    """
    mapped_y = pd.DataFrame(data=label_mapper(labels, selected_labels),
                            index=index, columns=['Phenotype'])
    y = mapped_y[mapped_y['Phenotype'].isin(selected_labels)]
    X = pd.DataFrame(data, index=index, columns=feature_names).loc[y.index]
    return datasets.base.Bunch(data=X.values, feature_names=X.columns,
                               target=y.values.ravel(), index=X.index.tolist())
