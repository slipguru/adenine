#!/usr/bin/python
# -*- coding: utf-8 -*-

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Federico Tomasi, Annalisa Barla
#
# FreeBSD License
######################################################################

import logging
# import numpy as np
from adenine.utils.extra import modified_cartesian, ensure_list, values_iterator

# from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering

from adenine.utils.extensions import DummyNone
from adenine.utils.extensions import Imputer
from adenine.utils.extensions import GridSearchCV
from adenine.utils.extensions import silhouette_score


def parse_preproc(key, content):
    """Parse the options of the preprocessing step.

    This function parses the preprocessing step coded as dictionary in the
    ade_config file.

    Parameters
    -----------
    key : {'None', 'Recenter', 'Standardize', 'Normalize', 'MinMax'}
        The type of selected preprocessing step.

    content : dict
        A dictionary containing parameters for each preprocessing
        class. Each parameter can be a list; in this case for each combination
        of parameters a different pipeline will be created.

    Returns
    -----------
    pptpl : tuple
        A tuple made like that ('preproc_name', preproc_obj), where preproc_obj
        is an sklearn 'transforms' (i.e. it has bot a .fit and .transform
        method).
    """
    if key.lower() == 'none':
        pp = DummyNone()
    elif key.lower() == 'recenter':
        pp = StandardScaler(with_mean=True, with_std=False)
    elif key.lower() == 'standardize':
        pp = StandardScaler(with_mean=True, with_std=True)
    elif key.lower() == 'normalize':
        content.setdefault('norm', 'l2')
        # pp = Normalizer(norm=content[1][0])
        pp = Normalizer(**content)
    elif key.lower() == 'minmax':
        content.setdefault('feature_range', (0, 1))
        pp = MinMaxScaler(**content)
    else:
        pp = DummyNone()
    return (key, pp)


def parse_dimred(key, content):
    """Parse the options of the dimensionality reduction step.

    This function does the same as parse_preproc but works on the
    dimensionality reduction & manifold learning options.

    Parameters
    -----------
    key : {'None', 'PCA', 'KernelPCA', 'Isomap', 'LLE', 'SE', 'MDS', 'tSNE'}
        The selected dimensionality reduction algorithm.

    content : dict
        A dictionary containing parameters for each dimensionality reduction
        class. Each parameter can be a list; in this case for each combination
        of parameters a different pipeline will be created.

    Returns
    -----------
    drtpl : tuple
        A tuple made like that ('dimres_name', dimred_obj), where dimred_obj is
        a sklearn 'transforms' (i.e. it has bot a .fit and .transform method).
    """
    drs = {'none': DummyNone, 'pca': PCA, 'incrementalpca': IncrementalPCA,
           'randomizedpca': RandomizedPCA, 'kernelpca': KernelPCA,
           'isomap': Isomap, 'lle': LocallyLinearEmbedding,
           'se': SpectralEmbedding, 'mds': MDS, 'tsne': TSNE}

    content.setdefault('n_components', 3)  # use three cluster as default
    dr = drs.get(key.lower(), DummyNone)(**content)
    return (key, dr)

    # if key.lower() == 'none':
    #     dr = DummyNone(**content)
    # elif key.lower() == 'pca':
    #     dr = PCA(**content) # this by default takes all the components it can
    # elif key.lower() == 'incrementalpca':
    #     dr = IncrementalPCA(**content)
    # elif key.lower() == 'randomizedpca':
    #     dr = RandomizedPCA(**content)
    # elif key.lower() == 'kernelpca':
    #     dr = KernelPCA(**content)
    # elif key.lower() == 'isomap':
    #     dr = Isomap(**content)
    # elif key.lower() == 'lle':
    #     dr = LocallyLinearEmbedding(**content)
    # elif key.lower() == 'ltsa':
    #     dr = LocallyLinearEmbedding(**content)
    # elif key.lower() == 'se':
    #     dr = SpectralEmbedding(**content)
    # elif key.lower() == 'mds':
    #     dr = MDS(**content)
    # elif key.lower() == 'tsne':
    #     dr = TSNE(**content)
    # else:
    #     dr = DummyNone(**content)
    # return (key, dr)


def parse_clustering(key, content):
    """Parse the options of the clustering step.

    This function does the same as parse_preproc but works on the clustering
    options.

    Parameters
    -----------
    key : {'KMeans', 'KernelKMeans', 'AP', 'MS', 'Spectral', 'Hierarchical'}
        The selected dimensionality reduction algorithm.

    content : dict
        A dictionary containing parameters for each clustering class.
        Each parameter can be a list; in this case for each combination
        of parameters a different pipeline will be created.

    Returns
    -----------
    cltpl : tuple
        A tuple made like that ('clust_name', clust_obj), where clust_obj
        implements the .fit method.
    """
    if 'auto' in [content.get('n_clusters', ''), content.get('preference', '')]:
        # Wrapper class that automatically detects the best number of clusters
        # via 10-Fold CV
        content.pop('n_clusters', '')
        content.pop('preference', '')

        kwargs = {'param_grid': [], 'n_jobs': -1,
                  'scoring': silhouette_score, 'cv': 10}

        if key.lower() == 'kmeans':
            content.setdefault('init', 'k-means++')
            content.setdefault('n_jobs', 1)
            kwargs['estimator'] = KMeans(**content)
        elif key.lower() == 'ap':
            kwargs['estimator'] = AffinityPropagation(**content)
            kwargs['affinity'] = kwargs['estimator'].affinity
        else:
            logging.warning("n_clusters = 'auto' specified outside kmeans or ap."
                            " Creating GridSearchCV pipeline anyway ...")
        cl = GridSearchCV(**kwargs)

    else:
        if key.lower() == 'kmeans':
            content.setdefault('n_jobs', -1)
            cl = KMeans(**content)
        elif key.lower() == 'ap':
            content.setdefault('preference', 1)
            cl = AffinityPropagation(**content)
        elif key.lower() == 'ms':
            cl = MeanShift(**content)
        elif key.lower() == 'spectral':
            cl = SpectralClustering(**content)
        elif key.lower() == 'hierarchical':
            cl = AgglomerativeClustering(**content)
        else:
            cl = DummyNone()
    return (key, cl)


def parse_steps(steps, max_n_pipes=100):
    """Parse the steps and create the pipelines.

    This function parses the steps coded as dictionaries in the ade_config
    files and creates a sklearn pipeline objects for each combination of
    imputing -> preprocessing -> dimensinality reduction -> clustering
    algorithms.

    A typical step may be of the following form:
        stepX = {'Algorithm': [On/Off flag, [variant0, ...]]}
    where On/Off flag = {True, False} and variantX = 'string'.

    Parameters
    -----------
    steps : list of dictionaries
        A list of (usually 4) dictionaries that contains the details of the
        pipelines to implement.

    max_n_pipes : int, optional, default: 100
        The maximum number of combinations allowed. This avoids a too expensive
        computation.

    Returns
    -----------
    pipes : list of sklearn.pipeline.Pipeline
        The returned list must contain every possible combination of
        imputing -> preprocessing -> dimensionality reduction -> clustering
        algorithms (up to max_n_pipes).
    """
    pipes = []       # a list of list of tuples input of sklearn Pipeline
    imputing, preproc, dimred, clustering = steps[:4]

    # Parse the imputing options
    i_lst_of_tpls = []
    if imputing['Impute'][0]:  # On/Off flag
        if len(imputing['Impute']) > 1:
            content_d = imputing['Impute'][1]
            content_values = values_iterator(content_d)
            for ll in modified_cartesian(*map(ensure_list, list(content_values))):
                content = {__k: __v for __k, __v in zip(list(content_d), ll)}
                i_lst_of_tpls.append(("Impute", Imputer(**content)))
        else:
            i_lst_of_tpls.append(("Impute", Imputer()))

    # Parse the preprocessing options
    pp_lst_of_tpls = []
    for key in preproc.keys():
        if preproc[key][0]:  # On/Off flag
            if len(preproc[key]) > 1:
                content_d = preproc[key][1]
                content_values = values_iterator(content_d)
                for ll in modified_cartesian(*map(ensure_list, list(content_values))):
                    content = {__k: __v for __k, __v in zip(list(content_d), ll)}
                    pp_lst_of_tpls.append(parse_preproc(key, content))
            else:
                pp_lst_of_tpls.append(parse_preproc(key, {}))
                # pp_lst_of_tpls.append(parse_preproc(key, preproc[key]))

    # Parse the dimensionality reduction & manifold learning options
    dr_lst_of_tpls = []
    for key in dimred.keys():
        if dimred[key][0]:  # On/Off flag
            if len(dimred[key]) > 1:
                content_d = dimred[key][1]
                content_values = values_iterator(content_d)
                for ll in modified_cartesian(*map(ensure_list, list(content_values))):
                    content = {__k: __v for __k, __v in zip(list(content_d), ll)}
                    dr_lst_of_tpls.append(parse_dimred(key, content))
            else:
                dr_lst_of_tpls.append(parse_dimred(key, {}))

    # Parse the clustering options
    cl_lst_of_tpls = []
    for key in clustering.keys():
        if clustering[key][0]:  # On/Off flag
            if len(clustering[key]) > 1:  # Discriminate from just flag or flag + args
                content_d = clustering[key][1]
                content_values = values_iterator(content_d)
                for ll in modified_cartesian(*map(ensure_list, list(content_values))):
                    content = {_k: _v for _k, _v in zip(list(content_d), ll)}
                    if not (content.get('affinity', '') in ['manhattan', 'precomputed'] and content.get('linkage', '') == 'ward'):
                        cl_lst_of_tpls.append(parse_clustering(key, content))

            else:  # just flag case
                cl_lst_of_tpls.append(parse_clustering(key, {}))


    #  Generate the list of list of tuples (i.e. the list of pipelines)
    pipes = modified_cartesian(i_lst_of_tpls, pp_lst_of_tpls, dr_lst_of_tpls,
                               cl_lst_of_tpls, pipes_mode=True)
    for pipe in pipes:
        logging.info("Generated pipeline: \n {} \n".format(pipe))
    logging.info("*** {} pipeline(s) generated ***".format(len(pipes)))

    #  Get only the first max_n_pipes
    if len(pipes) > max_n_pipes:
        logging.warning("Maximum number of pipelines reached. "
                        "I'm keeping the first {}".format(max_n_pipes))
        pipes = pipes[0:max_n_pipes]

    return pipes
