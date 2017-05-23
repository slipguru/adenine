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
from six.moves import filter


def get_GEO(accession_number, phenotype_name='title', return_gse=False):
    """Get the GEO data from its accession number.

    Parameters
    -----------
    accession_number : string
        'GSEXXXXX' is any GEO accession ID loaded by `GEOparse`.

    Returns
    -----------
    data : sklearn.datasets.base.Bunch
        the dataset bunch
    gse : GEOparse.GEOTypes.GSE
        the GEOparse object
    """
    gse = GEOparse.get_GEO(geo=accession_number, destdir=os.curdir,
                           silent=True, include_data=True,
                           how='full')
    xx = gse.pivot_samples('VALUE').transpose()
    index = xx.index.tolist()
    feature_names = xx.columns.tolist()
    yy = gse.phenotype_data[phenotype_name]
    data = datasets.base.Bunch(data=xx.values, target=yy.values,
                               feature_names=feature_names,
                               index=index)


    print('* Desired labels can be found with --phenotype_name = ')
    for k in gse.phenotype_data.keys():
        print('\t{}'.format(k))

    out = [data]
    if return_gse:
        out.append(gse)

    return out


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

def id2gs(data, gse):
    """Convert IDs into GENE_SYMBOL.

    Parameters
    -----------
    data : sklearn.datasets.base.Bunch
        the dataset bunch
    gse : GEOparse.GEOTypes.GSE
        the GEOparse object

    Returns
    -----------
    data : sklearn.datasets.base.Bunch
        where feature_names has the gene symbols
    """
    # Get the platform name
    platform = gse.gpls.keys()[0]

    # Create the lookup table
    lookup_table = pd.DataFrame(data=gse.gpls[platform].table['GENE_SYMBOL'].values,
                                index=gse.gpls[platform].table['ID'].values,
                                columns=['GENE_SYMBOL'])
    # Correct NaN failures
    for i, lt_value in enumerate(lookup_table.values.ravel()):
        if pd.isnull(lt_value):
            lookup_table.values[i] = str(lookup_table.index[i])+'__NO-MATCH'
    gene_symbol = [lookup_table['GENE_SYMBOL'].loc[_id] for _id in data.feature_names]

    # Make bunch and return
    return datasets.base.Bunch(data=data.data, feature_names=gene_symbol,
                               target=data.target, index=data.index)


def restrict_to_signature(data, signature):
    """Restrict the data to the genes in the signature.

    Parameters
    -----------
    data : sklearn.datasets.base.Bunch
        the dataset bunch
    signature : list
        list of signature genes

    Returns
    -----------
    data : sklearn.datasets.base.Bunch
        where feature_names has the gene symbols restricted to signature
    """
    df = pd.DataFrame(data=data.data, index=data.index,
                      columns=data.feature_names)
    # Filter out signatures gene not in the gene set
    signature = list(filter(lambda x: x in data.feature_names, signature))
    df = df[signature]
    # Make bunch and return
    return datasets.base.Bunch(data=df.values, feature_names=df.columns,
                               target=data.target, index=data.index)
