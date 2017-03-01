#!/usr/bin/python
# -*- coding: utf-8 -*-

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Federico Tomasi, Annalisa Barla
#
# FreeBSD License
######################################################################

import inspect
import logging

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.neural_network import BernoulliRBM
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering

from adenine.utils.extensions import DummyNone
from adenine.utils.extensions import Imputer
from adenine.utils.extensions import GridSearchCV
from adenine.utils.extensions import KernelPCA
from adenine.utils.extensions import silhouette_score

from adenine.utils.extra import modified_cartesian
from adenine.utils.extra import ensure_list
from adenine.utils.extra import values_iterator


def parse_imputing(key, content):
    """Parse the options of the imputing step.

    This function parses the imputing step coded as dictionary in the
    ade_config file.

    Parameters
    -----------
    key : class or str, like {'Impute', 'None'}
        The type of selected imputing step. In case in which key
        is a `class`, it must contain both a `fit` and `transform` method.

    content : dict
        A dictionary containing parameters for each imputing
        class. Each parameter can be a list; in this case for each combination
        of parameters a different pipeline will be created.

    Returns
    -----------
    tpl : tuple
        A tuple made like ('imputing_name', imputing_obj, 'imputing'),
        where imputing_obj is an sklearn 'transforms' (i.e. it has bot a fit
        and transform method).
    """
    if inspect.isclass(key):
        pi = key(**content)
        key = pi.__class__.__name__.lower()
    else:
        imputing_methods = {'none': DummyNone, 'impute': Imputer}
        pi = imputing_methods.get(key.lower(), DummyNone)(**content)
    return (key, pi, 'imputing')


def parse_preproc(key, content):
    """Parse the options of the preprocessing step.

    This function parses the preprocessing step coded as dictionary in the
    ade_config file.

    Parameters
    -----------
    key : class or str, like {'None', 'Recenter', 'Standardize', 'Normalize',
                              'MinMax'}
        The selected preprocessing algorithm. In case in which key
        is a `class`, it must contain both a `fit` and `transform` method.

    content : dict
        A dictionary containing parameters for each preprocessing
        class. Each parameter can be a list; in this case for each combination
        of parameters a different pipeline will be created.

    Returns
    -----------
    tpl : tuple
        A tuple made like ('preproc_name', preproc_obj, 'preproc'), where
        preproc_obj is an sklearn 'transforms' (i.e. it has bot a fit and
        transform method).
    """
    if inspect.isclass(key):
        pp = key(**content)
        key = pp.__class__.__name__.lower()
    elif key.lower() == 'none':
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
    return (key, pp, 'preproc')


def parse_dimred(key, content):
    """Parse the options of the dimensionality reduction step.

    This function does the same as parse_preproc but works on the
    dimensionality reduction & manifold learning options.

    Parameters
    -----------
    key : class or str, like {'None', 'PCA', 'KernelPCA', 'Isomap', 'LLE',
                              'SE', 'MDS', 'tSNE', 'RBM'}
        The selected dimensionality reduction algorithm. In case in which key
        is a `class`, it must contain both a `fit` and `transform` method.

    content : dict
        A dictionary containing parameters for each dimensionality reduction
        class. Each parameter can be a list; in this case for each combination
        of parameters a different pipeline will be created.

    Returns
    -----------
    tpl : tuple
        A tuple made like ('dimres_name', dimred_obj, 'dimred'), where
        dimred_obj is a sklearn 'transforms' (i.e. it has bot a .fit and .transform method).
    """
    if inspect.isclass(key):
        dr = key(**content)
        key = dr.__class__.__name__.lower()
    else:
        drs = {'none': DummyNone, 'pca': PCA, 'incrementalpca': IncrementalPCA,
               'randomizedpca': RandomizedPCA, 'kernelpca': KernelPCA,
               'isomap': Isomap, 'lle': LocallyLinearEmbedding,
               'se': SpectralEmbedding, 'mds': MDS, 'tsne': TSNE,
               'rbm': BernoulliRBM}

        content.setdefault('n_components', 3)  # use three cluster as default
        dr = drs.get(key.lower(), DummyNone)(**content)
    return (key, dr, 'dimred')


def parse_clustering(key, content):
    """Parse the options of the clustering step.

    This function does the same as parse_preproc but works on the clustering
    options.

    Parameters
    -----------
    key : class or str, like {'KMeans', 'AP', 'MS', 'Spectral', 'Hierarchical'}
        The selected clustering algorithm. In case in which key
        is a `class`, it must contain a `fit` method.

    content : dict
        A dictionary containing parameters for each clustering class.
        Each parameter can be a list; in this case for each combination
        of parameters a different pipeline will be created.

    Returns
    -----------
    tpl : tuple
        A tuple made like ('clust_name', clust_obj, 'clustering'), where
        clust_obj implements the `fit` method.
    """
    if inspect.isclass(key):
        cl = key(**content)
        key = cl.__class__.__name__.lower()

    elif 'auto' in (content.get('n_clusters', ''),
                    content.get('preference', '')) \
            and key.lower() != 'hierarchical':
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
            logging.error("n_clusters = 'auto' specified outside kmeans or "
                          "ap. Trying to create GridSearchCV pipeline anyway "
                          " ...")
        cl = GridSearchCV(**kwargs)
    elif 'auto' in (content.get('n_clusters', ''),
                    content.get('preference', '')) \
            and key.lower() == 'hierarchical':
        # TODO implement this
        # from adenine.utils.extensions import AgglomerativeClustering
        cl = AgglomerativeClustering(**content)
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
    return (key, cl, 'clustering')


def _lst_of_tpls(step, parsing_function, filt=None):
    """Generate a list of tuples for each parameter combination."""
    lst = []
    for key in step:
        if step[key][0]:  # On/Off flag
            if len(step[key]) > 1:
                content_d = step[key][1]
                content_vals = list(values_iterator(content_d))
                for ll in modified_cartesian(*map(ensure_list, content_vals)):
                    content = dict(zip(list(content_d), ll))
                    if filt is not None and filt(content):
                        continue
                    lst.append(parsing_function(key, content))
            else:
                lst.append(parsing_function(key, {}))
    return lst


def parse_steps(steps, max_n_pipes=200):
    """Parse the steps and create the pipelines.

    This function parses the steps coded as dictionaries in the ade_config
    files and creates a sklearn pipeline objects for each combination of
    imputing -> preprocessing -> dimensionality reduction -> clustering
    algorithms.

    A typical step may be of the following form:
        stepX = {'Algorithm': [On/Off flag, {'parameter1', [list of params]}]}
    where On/Off flag = {True, False} and 'list of params' allows to specify
    multiple params. In case in which the 'list of params' is actually a list,
    multiple pipelines are created for each combination of parameters.

    Parameters
    -----------
    steps : list of dictionaries
        A list of (usually 4) dictionaries that contains the details of the
        pipelines to implement.

    max_n_pipes : int, optional, default: 200
        The maximum number of combinations allowed. This avoids a too expensive
        computation.

    Returns
    -----------
    pipes : list of sklearn.pipeline.Pipeline
        The returned list must contain every possible combination of
        imputing -> preprocessing -> dimensionality reduction -> clustering
        algorithms (up to max_n_pipes).
    """
    im_lst_of_tpls = _lst_of_tpls(steps[0], parse_imputing)
    pp_lst_of_tpls = _lst_of_tpls(steps[1], parse_preproc)
    dr_lst_of_tpls = _lst_of_tpls(steps[2], parse_dimred)

    # When parsing clustering options, take care of error-generating parameters
    cl_lst_of_tpls = _lst_of_tpls(
        steps[3], parse_clustering,
        filt=(lambda x: x.get('affinity', '') in ['manhattan', 'precomputed']
              and x.get('linkage', '') == 'ward'))

    # Generate the list of list of tuples (i.e. the list of pipelines)
    pipes = modified_cartesian(im_lst_of_tpls, pp_lst_of_tpls, dr_lst_of_tpls,
                               cl_lst_of_tpls, pipes_mode=True)
    for pipe in pipes:
        logging.info("Generated pipeline: \n %s \n", pipe)
    logging.info("*** %d pipeline(s) generated ***", len(pipes))

    #  Get only the first max_n_pipes
    if len(pipes) > max_n_pipes:
        logging.warning("Maximum number of pipelines reached. "
                        "I'm keeping the first %d", max_n_pipes)
        pipes = pipes[:max_n_pipes]

    return pipes
