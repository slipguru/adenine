#!/usr/bin/python -W ignore::DeprecationWarning
# -*- coding: utf-8 -*-

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Federico Tomasi, Annalisa Barla
#
# FreeBSD License
######################################################################

import os
import logging
import cPickle as pkl
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('AGG')
import seaborn as sns
import multiprocessing as mp

from sklearn import metrics

from adenine.core import plotting
from adenine.core.plotting import *
from adenine.utils.extra import title_from_filename
from adenine.utils.extra import timed, items_iterator

GLOBAL_INFO = ''  # to save info before logging is loaded


def est_clst_perf(root, data_in, labels=None, t_labels=None, model=(),
                  metric='euclidean'):
    """Estimate the clustering performance.

    This function estimate the clustering performance by means of several
    indexes. Then saves the results in a tree-like structure in the root folder.

    Parameters
    -----------
    root : string
        The root path for the output creation.

    data_in : array of float, shape : (n_samples, n_dimensions)
        The low space embedding estimated by the dimensinality reduction and
        manifold learning algorithm.

    labels : array of float, shape : n_samples
        The label assignment performed by the clustering algorithm.

    t_labels : array of float, shape : n_samples
        The true label vector; None if missing.

    model : sklearn or sklearn-like object
        An instance of the class that evaluates a step. In particular this must
        be a clustering model provided with the clusters_centers_ attribute
        (e.g. KMeans).

    metric : string
        The metric used during the clustering algorithms.
    """
    perf_out = dict()
    try:
        if hasattr(model, 'inertia_'):
            # Sum of distances of samples to their closest cluster center.
            perf_out['inertia'] = model.inertia_

        perf_out['silhouette'] = metrics.silhouette_score(data_in, labels, metric=metric)
        if t_labels is not None:
            # the next indexes need a gold standard
            perf_out['ari'] = metrics.adjusted_rand_score(t_labels, labels)
            perf_out['ami'] = metrics.adjusted_mutual_info_score(t_labels, labels)
            perf_out['homogeneity'] = metrics.homogeneity_score(t_labels, labels)
            perf_out['completeness'] = metrics.completeness_score(t_labels, labels)
            perf_out['v_measure'] = metrics.v_measure_score(t_labels, labels)

    except ValueError as e:
        logging.info("Clustering performance evaluation failed for {}. "
                     "Error: {}".format(model, e))
        # perf_out = {'empty': 0.0}
        perf_out['###'] = 0.

    # Define the filename
    filename = os.path.join(root, os.path.basename(root))
    with open(filename+'_scores.txt', 'w') as f:
        f.write("------------------------------------\n")
        f.write("Adenine: Clustering Performance for \n")
        f.write("\n")
        f.write(title_from_filename(root, " --> ") + "\n")
        f.write("------------------------------------\n")
        f.write("Index Name{}|{}Index Score\n".format(' '*10, ' '*4))
        f.write("------------------------------------\n")
        for elem in sorted(perf_out.keys()):
            f.write("{}{}|{}{:.4}\n"
                    .format(elem, ' '*(20-len(elem)), ' '*4, perf_out[elem]))
            f.write("------------------------------------\n")

    # pkl Dump
    with open(filename+'_scores.pkl', 'w+') as f:
        pkl.dump(perf_out, f)
    logging.info("Dumped : {}".format(filename+'_scores.pkl'))


def get_step_attributes(step, pos):
    """Get the attributes of the input step.

    This function returns the attributes (i.e. level, name, outcome) of the
    input step. This comes handy when dealing with steps with more than one
    parameter (e.g. KernelPCA 'poly' or 'rbf').

    Parameters
    -----------
    step : list
        A step coded by ade_run.py as
        [name, level, param, data_out, data_in, mdl obj, voronoi_mdl_obj]

    pos : int
        The position of the step inside the pipeline.

    Returns
    -----------
    name : string
        A unique name for the step (e.g. KernelPCA_rbf).

    level : {imputing, preproc, dimred, clustering}
        The step level.

    data_out : array of float, shape : (n_samples, n_out)
        Where n_out is n_dimensions for dimensionality reduction step, or 1
        for clustering.

    data_in : array of float, shape : (n_samples, n_in)
        Where n_in is n_dimensions for preprocessing/imputing/dimensionality
        reduction step, or n_dim for clustering (because the data have already
        been dimensionality reduced).

    param : dictionary
        The parameters of the sklearn object implementing the algorithm.

    mdl_obj : sklearn or sklearn-like object
        This is an instance of the class that evaluates a step.
    """
    name = step[0]
    level = step[1]  # {imputing, preproc, dimred, clustering}
    param = step[2]
    data_out = step[3]
    data_in = step[4]
    mdl_obj = step[5]
    voronoi_mdl_obj = step[6]
    if level.lower() == 'none':
        if pos == 0:
            level = 'preproc'
        elif pos == 1:
            level = 'dimred'
    metric = 'euclidean'

    # Imputing level
    if param.get('missing_values', ''):
        name += '-'+param['missing_values']
    if param.get('strategy', ''):
        name += '_'+param['strategy']

    # Preprocessing level
    if param.get('norm', ''):  # normalize
        name += '_'+param['norm']
    elif param.get('feature_range', ''):  # minmax
        name += "_({} - {})".format(*param['feature_range'])

    # Append additional parameters in the step name
    if name == 'KernelPCA':
        name += '_'+param['kernel']
    elif name == 'LLE':
        name += '_'+param['method']
    elif name == 'MDS':
        if param['metric']:
            name += '_metric'
        else:
            name += '_nonmetric'
    elif name == 'Hierarchical':
        name += '_'+param['affinity']+'_'+param['linkage']

    if param.get('affinity', '') == 'precomputed':
        metric = 'precomputed'

    # n_clusters = param.get('n_clusters', 0) or param.get('best_estimator_', dict()).get('n_clusters', 0) or param.get('best_estimator_', dict()).get('clusters_centers_', np.array([])).shape[0]
    n_clusters = param.get('n_clusters', 0) or  \
                 param.get('best_estimator_', dict()).get('cluster_centers_',
                           np.empty(0)).shape[0] or \
                 param.get('cluster_centers_', np.empty(0)).shape[0] or \
                 mdl_obj.__dict__.get('cluster_centers_', np.empty(0)).shape[0]
    if n_clusters > 0:
        name += '_' + str(n_clusters) + '-clusts'

    logging.info("{} : {}".format(level, name))
    return name, level, param, data_out, data_in, mdl_obj, voronoi_mdl_obj, metric


def make_df_clst_perf(root):
    """Summarize all the clustering performance estimations.

    Given the output file produced by est_clst_perf(), this function groups all
    of them together in friendly text and latex files, and saves the two files
    produced in a tree-like structure in the root folder.

    Parameters
    -----------
    root : string
        The root path for the output creation.
    """
    measures = ('ami', 'ari', 'completeness', 'homogeneity', 'v_measure',
                'inertia', 'silhouette')
    df = pd.DataFrame(columns=['pipeline']+list(measures))
    for root, directories, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith('_scores.pkl'):
                with open(os.path.join(root, fn), 'r') as f:
                    perf_out = pkl.load(f)
                perf_out['pipeline'] = title_from_filename(root, step_sep=" --> ")
                df = df.append(perf_out, ignore_index=True)
    df = df.fillna('')

    pipe_header = 'preprocess --> dim red --> clustering'
    size_pipe = max([len(p) for p in df['pipeline']]+[len(pipe_header)])
    size_ami, size_ari, size_com, size_hom, \
    size_vme, size_ine, size_sil = [2 +
        max([len('{: .3}'.format(p)) if p != '' else 3 for p in df[__]] + [len(__)])
            for __ in measures]

    # find the best value for each score
    best_scores = {__: max([p for p in df[__] if p != ''] or [np.nan]) for __ in measures}

    with open(os.path.join(root,'summary_scores.txt'), 'w') as f, \
         open(os.path.join(root,'summary_scores.tex'), 'w') as g:
        header = "{}{}|{}ami  |{}ari  |{}completeness  |{}homogeneity  |{}v_measure  |{}inertia  |{}silhouette  \n" \
            .format(pipe_header, ' '*(size_pipe-len(pipe_header)),
                    ' '*(size_ami-5), ' '*(size_ari-5),
                    ' '*(size_com-len("completeness  ")),
                    ' '*(size_hom-len("homogeneity  ")),
                    ' '*(size_vme-len("v_measure  ")),
                    ' '*(size_ine-len("inertia  ")),
                    ' '*(size_sil-len("silhouette  ")))
        f.write("-"*len(header) + "\n")
        f.write("Adenine: Clustering Performance for each pipeline\n")
        f.write("-"*len(header) + "\n")
        f.write(header)
        f.write("-"*len(header) + "\n")

        g.write(r"\documentclass{article}" "\n"
                r"\usepackage{adjustbox}" "\n"
                r"\usepackage{caption}" "\n"
                r"\captionsetup[table]{skip=10pt}" "\n"
                r"\begin{document}" "\n"
                r"\begin{table}[h!]" "\n"
                r"\centering" "\n"
                r"\caption{Adenine: Clustering Performance for each pipeline}" "\n"
                r"\label{clust-perf}" "\n"
                r"\begin{adjustbox}{max width=\textwidth}" "\n"
                r"\begin{tabular}{l|rc|rc|rc|rc|rc|rc|rc}" "\n"
                r"\textbf{preprocess $\to$ dim red $\to$ clustering} & \textbf{ami} "
                r"&& \textbf{ari} && \textbf{completeness} && \textbf{homogeneity} "
                r"&& \textbf{v\_measure} && \textbf{inertia} && \textbf{silhouette}"
                r" & \\ \hline" "\n")

        for _ in df.iterrows():
            row = _[1]
            ami, ari, com, hom, vme, ine, sil = ['{: .3}'.format(row[__])
                                if row[__] != '' else '---' for __ in measures]

            star = {__ : ' *' if row[__] == best_scores[__] else '  ' for __ in measures}
            f.write("{}{}|{}{}{}|{}{}{}|{}{}{}|{}{}{}|{}{}{}|{}{}{}|{}{}{}\n"
                 .format(row['pipeline'],' '*(size_pipe-len(row['pipeline'])),
                 ' '*(abs(size_ami-len(str(ami))-2)), ami, star['ami'],
                 ' '*(abs(size_ari-len(str(ari))-2)), ari, star['ari'],
                 ' '*(abs(size_com-len(str(com))-2)), com, star['completeness'],
                 ' '*(abs(size_hom-len(str(hom))-2)), hom, star['homogeneity'],
                 ' '*(abs(size_vme-len(str(vme))-2)), vme, star['v_measure'],
                 ' '*(abs(size_ine-len(str(ine))-2)), ine, star['inertia'],
                 ' '*(abs(size_sil-len(str(sil))-2)), sil, star['silhouette']
            ))

            g.write(r"{} & {}&{} & {}&{} & {}&{} & {}&{} & {}&{} & {}&{} & {}&{} \\" "\n"
                .format(
                row['pipeline'].replace('-->', r'$\to$'),
                ami, star['ami'],
                ari, star['ari'],
                com, star['completeness'],
                hom, star['homogeneity'],
                vme, star['v_measure'],
                ine, star['inertia'],
                sil, star['silhouette'],
            ))

        f.write("-"*len(header) + "\n")
        g.write(r"\hline" "\n"
                r"\end{tabular}" "\n"
                r"\end{adjustbox}" "\n"
                r"\end{table}" "\n"
                r"\end{document}")
    # df.to_csv(os.path.join(rootFolder,'all_scores.csv'), na_rep='-', index_label=False, index=False)


def analysis_worker(elem, root, y, feat_names, class_names, lock):
    """Parallel pipelines analysis.

    Parameters
    ----------
    elem : list
        The first two element of this list are the pipe_id and all the data of
        that pipeline.

    root : string
        The root path for the output creation.

    y : array of float, shape : n_samples
        The label vector; None if missing.

    feat_names : array of integers (or strings), shape : n_features
        The feature names; a range of numbers if missing.

    class_names : array of integers (or strings), shape : n_features
        The class names; a range of numbers if missing.

    lock : multiprocessing.synchronize.Lock
        Obtained by multiprocessing.Lock().
        Needed for optional creation of directories.
    """
    # Getting pipeID and content
    pipe, content = elem[:2]

    out_folder = ''  # where the results will be placed
    logging.info("------\n{} : \n".format(pipe))
    for i, step in enumerate(sorted(content.keys())):
        # Tree-like folder structure definition
        step_name, step_level, step_param, step_out, step_in, mdl_obj, \
            voronoi_mdl_obj, metric = get_step_attributes(content[step], pos=i)

        # Output folder definition & creation
        out_folder = os.path.join(out_folder, step_name)
        rootname = os.path.join(root, out_folder)
        with lock:
            if not os.path.exists(rootname):
                os.makedirs(rootname)

        # Launch analysis
        if step_level == 'dimred':
            make_scatter(root=rootname, data_in=step_out, labels=y, true_labels=True)
            make_silhouette(root=rootname, labels=y, data_in=step_out, model=mdl_obj)

            if hasattr(mdl_obj, 'explained_variance_ratio_'):
                plot_PCmagnitude(root=rootname,
                                 points=mdl_obj.explained_variance_ratio_,
                                 title='Explained variance ratio')
            if hasattr(mdl_obj, 'lambdas_'):
                plot_PCmagnitude(root=rootname,
                                 points=mdl_obj.lambdas_/np.sum(mdl_obj.lambdas_),
                                 title='Normalized eigenvalues of the centered kernel matrix')
        if step_level == 'clustering':
            if hasattr(mdl_obj, 'affinity_matrix_'):
                n_clusters = mdl_obj.__dict__.get('cluster_centers_', np.empty(0)).shape[0]
                if hasattr(mdl_obj, 'n_clusters'):
                    n_clusters = mdl_obj.n_clusters

                plot_eigs(root=rootname, affinity=mdl_obj.affinity_matrix_,
                          n_clusters=n_clusters,
                          title='Eigenvalues of the graph associated to the affinity matrix')
            if hasattr(mdl_obj, 'cluster_centers_'):
                _est_name = mdl_obj.__dict__.get('estimator_name', '') or type(mdl_obj).__name__
                if _est_name != 'AffinityPropagation':
                    # disable the voronoi plot for affinity prop
                    make_voronoi(root=rootname, labels=y, data_in=step_in,
                                 model=voronoi_mdl_obj)
            elif hasattr(mdl_obj, 'n_leaves_'):
                make_tree(root=rootname, data_in=step_in, labels=step_out, model=mdl_obj)
                make_dendrogram(root=rootname, data_in=step_in, labels=y, model=mdl_obj)

            make_scatter(root=rootname, labels=step_out, data_in=step_in, model=mdl_obj)
            make_silhouette(root=rootname, labels=step_out, data_in=step_in, model=mdl_obj)
            est_clst_perf(root=rootname, data_in=step_in, labels=step_out,
                          t_labels=y, model=mdl_obj, metric=metric)


@timed
def analyze(input_dict, root, y=None, feat_names=(), class_names=(), **kwargs):
    """Analyze the results of ade_run.

    This function analyze the dictionary generated by ade_run, generates the
    plots, and saves them in a tree-like folder structure in rootFolder.

    Parameters
    -----------
    input_dict : dictionary
        The dictionary created by ade_run.py on some data.

    root : string
        The root path for output creation.

    y : array of float, shape : n_samples
        The label vector; None if missing.

    feat_names : array of integers (or strings), shape : n_features
        The feature names; a range of numbers if missing.

    class_names : array of integers (or strings), shape : n_features
        The class names; a range of numbers if missing.

    kwargs : dictionary
        Additional optional parameters. In particular it can contain
        'plotting_context' and 'file_format' variables, if specified in
        the config file.
    """
    if GLOBAL_INFO:
        logging.info(GLOBAL_INFO)
    if kwargs.get('plotting_context', None):
        sns.set_context(kwargs.get('plotting_context'))

    FILE_FORMATS = ('png', 'pdf')
    ff = kwargs.get('file_format', FILE_FORMATS[0]).lower()

    if ff not in FILE_FORMATS:
        logging.warning("File format unknown. "
                        "Please select one of {}".format(FILE_FORMATS))
        plotting.set_file_ext(FILE_FORMATS[0])
    else:
        plotting.set_file_ext(ff)
    logging.info("File format set to {}".format(plotting.GLOBAL_FF))
    lock = mp.Lock()
    # Parallel(n_jobs=len(inputDict))(delayed(analysis_worker)(elem,rootFolder,y,feat_names,class_names,lock) for elem in inputDict.iteritems())
    ps = []
    for elem in items_iterator(input_dict):
        p = mp.Process(target=analysis_worker,
                       args=(elem, root, y, feat_names, class_names, lock))
        p.start()
        ps.append(p)

    for p in ps:
        p.join()

    make_df_clst_perf(root)
