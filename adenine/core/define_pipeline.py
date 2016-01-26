#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
import numpy as np
from adenine.utils.extra import modified_cartesian

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
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

class DummyNone:
    """Dummy class that does nothing.

    It is an sklearn 'transforms', it implements both a fit and a transform method and it just returns the data in input. It has been created only for consistency with sklearn.
    """
    def __init__(self):
        pass

    def fit(self,X):
        return self

    def transform(self,X):
        return X

    def get_params(self):
        return dict()

def parse_preproc(key, content):
        """Parse the options of the preprocessing step.

        This function parses the preprocessing step coded as dictionary in the ade_config file.

        Parameters
        -----------
        key : {'None', 'Recenter', 'Standardize', 'Normalize', 'MinMax'}
            The type of selected preprocessing step.

        content : list, len : 2
            A list containing the On/Off flag and a nested list of extra parameters (e.g. [min,max] for Min-Max scaling).

        Returns
        -----------
        pptpl : tuple
            A tuple made like that ('PreprocName', preprocObj), where preprocObj is an sklearn 'transforms' (i.e. it has bot a .fit and .transform method).
        """
        if key.lower() == 'none':
            pp = DummyNone()
        elif key.lower() == 'recenter':
            pp = StandardScaler(with_mean = True, with_std = False)
        elif key.lower() == 'standardize':
            pp = StandardScaler(with_mean = True, with_std = True)
        elif key.lower() == 'normalize':
            pp = Normalizer(norm = content[1][0])
        elif key.lower() == 'minmax':
            pp = MinMaxScaler(feature_range = (content[1][0], content[1][1]))
        else:
            pp = DummyNone()
        return (key, pp)

def parse_dimred(key, content):
    """Parse the options of the dimensionality reduction step.

    This function does the same as parse_preproc but works on the dimensionality reduction & manifold learning options.

    Parameters
    -----------
    key : {'None', 'PCA', 'KernelPCA', 'Isomap', 'LLE', 'SE', 'MDS', 'tSNE'}
        The selected dimensionality reduction algorithm.

    content : list, len : 2
        A list containing the On/Off flag and a nested list of extra parameters (e.g. ['rbf,'poly'] for KernelPCA).

    Returns
    -----------
    drtpl : tuple
        A tuple made like that ('DimRedName', dimredObj), where dimredObj is an sklearn 'transforms' (i.e. it has bot a .fit and .transform method).
    """
    if key.lower() == 'none':
        dr = DummyNone()
    elif key.lower() == 'pca':
        #dr = PCA(n_components = 'mle')
        dr = PCA(n_components = 2)
    elif key.lower() == 'incrementalpca':
        dr = IncrementalPCA()
    elif key.lower() == 'randomizedpca':
        dr = RandomizedPCA()
    elif key.lower() == 'kernelpca':
        dr = KernelPCA(kernel = content, n_components = None)
    elif key.lower() == 'isomap':
        dr = Isomap()
    elif key.lower() == 'lle':
        dr = LocallyLinearEmbedding(method = content)
    elif key.lower() == 'ltsa':
        dr = LocallyLinearEmbedding(method = content)
    elif key.lower() == 'se':
        dr = SpectralEmbedding()
    elif key.lower() == 'mds':
        if content == 'nonmetric':
            dr = MDS(metric = 'False')
        else:
            dr = MDS(metric = 'True')
    elif key.lower() == 'tsne':
        dr = TSNE()
    else:
        dr = DummyNone()
    return (key, dr)

def parse_clustering(key, content):
    """Parse the options of the clustering step.

    This function does the same as parse_preproc but works on the clustering options.

    Parameters
    -----------
    key : {'KMeans', 'KernelKMeans', 'AP', 'MS', 'Spectral', 'Hierarchical'}
        The selected dimensionality reduction algorithm.

    content : list, len : 2
        A list containing the On/Off flag and a nested list of extra parameters (e.g. ['rbf,'poly'] for KernelKMeans).

    Returns
    -----------
    cltpl : tuple
        A tuple made like that ('ClusteringName', clustObj), where clustObj implements the .fit method.
    """
    if key.lower() == 'kmeans':
        cl = KMeans(n_clusters = content, init = 'k-means++', n_jobs = -1)
    elif key.lower() == 'kernelkmeans': # TODO
        kernel = content[1]
        cl = KMeans(n_clusters = content[0], init = 'k-means++', n_jobs = -1)
    elif key.lower() == 'ap':
        cl = AffinityPropagation()
    elif key.lower() == 'ms':
        cl = MeanShift()
    elif key.lower() == 'spectral':
        cl = SpectralClustering(n_clusters = content)
    elif key.lower() == 'hierarchical':
        if len(content) > 2:
            cl = AgglomerativeClustering(n_clusters=content[0], affinity=content[1], linkage=content[2])
        else:
            cl = AgglomerativeClustering(n_clusters=content[0], affinity=content[1])

    else:
        cl = DummyNone()
    return (key, cl)

def parse_steps(steps):
    """Parse the steps and create the pipelines.

    This function parses the steps coded as dictionaries in the ade_config files and creates a sklearn pipeline objects for each combination of imputing -> preprocessing -> dimensinality reduction -> clustering algorithms.

    A typical step may be of the following form:
        stepX = {'Algorithm': [On/Off flag, [variant0, ...]]}
    where On/Off flag = {True, False} and variantX = 'string'.

    Parameters
    -----------
    steps : list of dictionaries
        A list of (usually 4) dictionaries that contains the details of the pipelines to implement.

    Returns
    -----------
    pipes : list of sklearn.pipeline.Pipeline
        The returned list must contain every possible combination of imputing -> preprocessing -> dimensionality reduction -> clustering algorithms. The maximum number of pipelines that could be generated is 20, even if the number of combinations is higher.
    """
    max_n_pipes = 100 # avoiding unclear (too-long) outputs
    pipes = []       # a list of list of tuples input of sklearn Pipeline

    imputing   = steps[0]
    preproc    = steps[1]
    dimred     = steps[2]
    clustering = steps[3]

    # Parse the imputing options
    i_lst_of_tpls = []
    if imputing['Impute'][0]: # On/Off flag
        for name in imputing['Replacement']:
            imp = Imputer(missing_values = imputing['Missing'][0],
                          strategy = name)
            i_lst_of_tpls.append(("Impute_"+name, imp))

    # Parse the preprocessing options
    pp_lst_of_tpls = []
    for key in preproc.keys():
        if preproc[key][0]: # On/Off flag
            pp_lst_of_tpls.append(parse_preproc(key, preproc[key]))

    # Parse the dimensionality reduction & manifold learning options
    dr_lst_of_tpls = []
    for key in dimred.keys():
        if dimred[key][0]: # On/Off flag
            if len(dimred[key]) > 1:# For each variant (e.g. 'rbf' or
                for k in dimred[key][1]: # 'poly' for KernelPCA)
                    dr_lst_of_tpls.append(parse_dimred(key, k))
            else:
                dr_lst_of_tpls.append(parse_dimred(key, dimred[key]))

     # Parse the clustering options
    cl_lst_of_tpls = []
    for key in clustering.keys():
        if clustering[key][0]: # On/Off flag
            if len(clustering[key]) > 1: # Discriminate from just flag or flag + args
                if len(clustering[key][1]) > 1: # discrimitate from 1 arg or 2+ args
                    if len(clustering[key][1]) > 2:
                        for k1, k2, k3 in modified_cartesian([clustering[key][1][0]], clustering[key][1][1], clustering[key][1][2]):
                            if k2 is 'precomputed':
                                cl_lst_of_tpls.append(parse_clustering(key, [k1,k2,'complete']))
                            elif not (k2 is 'manhattan' and k3 is 'ward'): # that doesn't work together
                                cl_lst_of_tpls.append(parse_clustering(key, [k1,k2,k3]))
                    else: # 2 args case
                        for k1, k2 in zip([clustering[key][1][0]], clustering[key][1][1]):
                            if k2 is 'precomputed':
                                cl_lst_of_tpls.append(parse_clustering(key, [k1,k2,'complete']))
                            else:
                                cl_lst_of_tpls.append(parse_clustering(key, [k1,k2]))
                else: # 1 arg case
                    for k in clustering[key][1]:
                        cl_lst_of_tpls.append(parse_clustering(key, k))
            else: # just flag case
                cl_lst_of_tpls.append(parse_clustering(key, clustering[key]))


    # Generate the list of list of tuples (i.e. the list of pipelines)
    pipes =  modified_cartesian(i_lst_of_tpls, pp_lst_of_tpls, dr_lst_of_tpls, cl_lst_of_tpls)
    for pipe in pipes:
        logging.info("Generated pipeline: \n {} \n".format(pipe))
    logging.info("*** {} pipeline(s) generated ***".format(len(pipes)))

    # Get only the first max_n_pipes
    if len(pipes) > max_n_pipes:
        logging.warning("Maximum number of pipelines reached. I'm keeping the first {}".format(max_n_pipes))
        pipes = pipes[0:max_n_pipes]

    return pipes
