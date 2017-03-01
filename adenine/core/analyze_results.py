#!/usr/bin/env python -W ignore::DeprecationWarning
# -*- coding: utf-8 -*-
"""Adenine analyzer module."""
######################################################################
# Copyright (C) 2016 Samuele Fiorini, Federico Tomasi, Annalisa Barla
#
# FreeBSD License
######################################################################

import os
import shutil
import logging
import matplotlib; matplotlib.use('AGG')
import multiprocessing as mp
import numpy as np
import pandas as pd
import seaborn as sns
import subprocess

try:
    import cPickle as pkl
except:
    import pickle as pkl

from sklearn import metrics

from adenine.core import plotting
from adenine.utils import scores
from adenine.utils.extra import title_from_filename
from adenine.utils.extra import timed, items_iterator

# to save info before logging is loaded
GLOBAL_INFO = 'matplotlib backend set to AGG'


def est_clst_perf(root, data_in, labels=None, t_labels=None, model=None,
                  metric='euclidean'):
    """Estimate the clustering performance.

    This estimates the clustering performance by means of several indexes.
    Results are saved in a tree-like structure in the root folder.

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

            perf_out['fscore'] = scores.precision_recall_fscore(
                scores.confusion_matrix(t_labels, labels)[0])[2]

    except ValueError as e:
        logging.warning("Clustering performance evaluation failed for %s. "
                        "Error: %s", model, str(e))
        # perf_out = {'empty': 0.0}
        perf_out['###'] = 0.

    # Define the filename
    filename = os.path.join(root, os.path.basename(root))
    with open(filename + '_scores.txt', 'w') as f:
        f.write("------------------------------------\n"
                "Adenine: Clustering Performance for \n"
                "\n" + title_from_filename(root, " --> ") + "\n"
                "------------------------------------\n")
        f.write("Index Name{}|{}Index Score\n".format(' ' * 10, ' ' * 4))
        f.write("------------------------------------\n")
        for elem in sorted(perf_out.keys()):
            f.write("{}{}|{}{:.4}\n"
                    .format(elem, ' ' * (20 - len(elem)), ' ' * 4,
                            perf_out[elem]))
            f.write("------------------------------------\n")

    # pkl Dump
    filename += '_scores.pkl'
    with open(filename, 'wb') as f:
        pkl.dump(perf_out, f)
    logging.info("Dumped : %s", filename)


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
                'inertia', 'silhouette', 'fscore')
    df = pd.DataFrame(columns=['pipeline'] + list(measures))
    for root_, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith('_scores.pkl'):
                with open(os.path.join(root_, fn), 'rb') as f:
                    perf_out = pkl.load(f)
                perf_out['pipeline'] = title_from_filename(root_,
                                                           step_sep=" --> ")
                df = df.append(perf_out, ignore_index=True)
    df = df.fillna('')
    nan_val = '---'

    pipe_header = 'preprocess --> dim red --> clustering'
    size_pipe = max([len(p) for p in df['pipeline']] + [len(pipe_header)])
    sizes = [3 + max([len('{: .3}'.format(p)) if p != '' else len(nan_val)
                     for p in df[mm]] + [len(mm)]) for mm in measures]

    # find the best value for each score
    best_scores = {
        mm: max([p for p in df[mm] if p != ''] or [np.nan]) for mm in measures}

    with open(os.path.join(root, 'summary_scores.txt'), 'w') as f, \
            open(os.path.join(root, 'summary_scores.tex'), 'w') as g:
        measures_header = [' ' * max(size - len(x) - 2, 1) + x + '  '
                           for size, x in zip(sizes, measures)]
        header = "{}{}|{}\n" \
                 .format(pipe_header,
                         ' ' * (size_pipe - len(pipe_header)),
                         '|'.join(measures_header))
        f.write("-" * len(header) + "\n")
        f.write("Adenine: Clustering Performance for each pipeline\n")
        f.write("-" * len(header) + "\n")
        f.write(header)
        f.write("-" * len(header) + "\n")

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
                r"\begin{tabular}{l|rc|rc|rc|rc|rc|rc|rc|rc}" "\n"
                r"\textbf{preprocess $\to$ dim red $\to$ clustering} & \textbf{ami} "
                r"&& \textbf{ari} && \textbf{completeness} && \textbf{homogeneity} "
                r"&& \textbf{v\_measure} && \textbf{inertia} && \textbf{silhouette} "
                r"&& \textbf{fscore}"
                r" & \\ \hline " "\n")

        for _ in df.iterrows():
            row = _[1]
            all_measures = ['{: .3}'.format(row[mm]) if row[mm] != ''
                            else nan_val for mm in measures]

            stars = [' *' if row[mm] == best_scores[mm] else '  ' for mm in measures]
            row_measure = [' ' * max(size - len(x) - 2, 1) + x + ss
                           for size, x, ss in zip(sizes, all_measures, stars)]
            f.write("{}{}|{}\n"
                    .format(
                        row['pipeline'],
                        ' ' * (size_pipe - len(row['pipeline'])),
                        '|'.join(row_measure)
                    ))
            row_tex = [x + r'&' + ss for x, ss in zip(all_measures, stars)]
            g.write(r"{} & {} \\" "\n"
                    .format(
                        row['pipeline'].replace('-->', r'$\to$'),
                        r'&'.join(row_tex)
                    ))

        f.write("-" * len(header) + "\n")
        g.write(r"\hline" "\n"
                r"\end{tabular}" "\n"
                r"\end{adjustbox}" "\n"
                r"\end{table}" "\n"
                r"\end{document}")


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
    -------
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
    name, level, param, data_out, \
        data_in, mdl_obj, voronoi_mdl_obj = step[:7]

    if level.lower() == 'none':
        if pos == 0:
            level = 'preproc'
        elif pos == 1:
            level = 'dimred'

    # Imputing level
    if param.get('missing_values', ''):
        name += '-' + param['missing_values']
    if param.get('strategy', ''):
        name += '_' + param['strategy']

    # Preprocessing level
    if param.get('norm', ''):  # normalize
        name += '_' + param['norm']
    elif param.get('feature_range', ''):  # minmax
        name += "_({} - {})".format(*param['feature_range'])

    # Append additional parameters in the step name
    if name == 'KernelPCA':
        name += '_' + param['kernel']
    elif name == 'LLE':
        name += '_' + param['method']
    elif name == 'MDS':
        if param['metric']:
            name += '_metric'
        else:
            name += '_nonmetric'
    elif name == 'Hierarchical':
        name += '_' + param['affinity'] + '_' + param['linkage']
    elif name == 'SE':
        name += '_' + param['affinity']

    try:
        n_clusters = param.get('n_clusters', 0) or  \
            param.get('best_estimator_', dict()).get('cluster_centers_',
                                                     np.empty(0)).shape[0] or \
            param.get('cluster_centers_', np.empty(0)).shape[0] or \
            mdl_obj.__dict__.get('n_clusters', 0) or \
            mdl_obj.__dict__.get('cluster_centers_', np.empty(0)).shape[0]
    except StandardError:
        n_clusters = 0
    if n_clusters > 0:
        name += '_' + str(n_clusters) + '-clusts'

    metric = param.get('affinity', None) or 'euclidean'
    return (name, level, param, data_out, data_in, mdl_obj,
            voronoi_mdl_obj, metric)


def analysis_worker(elem, root, y, feat_names, index, lock):
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

    index : list of integers (or strings)
        This is the samples identifier, if provided as first column (or row) of
        of the input file. Otherwise it is just an incremental range of size
        n_samples.

    lock : multiprocessing.synchronize.Lock
        Obtained by multiprocessing.Lock().
        Needed for optional creation of directories.
    """
    # Getting pipeID and content
    pipe, content = elem[:2]

    out_folder = ''  # where the results will be placed
    logging.info("Start {} --".format(pipe))
    for i, step in enumerate(sorted(content.keys())):
        # Tree-like folder structure definition
        step_name, step_level, step_param, step_out, step_in, mdl_obj, \
            voronoi_mdl_obj, metric = get_step_attributes(content[step], pos=i)
        logging.info("LEVEL {} : {}".format(step_level, step_name))

        # Output folder definition & creation
        out_folder = os.path.join(out_folder, step_name)
        rootname = os.path.join(root, out_folder)
        with lock:
            if not os.path.exists(rootname):
                os.makedirs(rootname)

        # Launch analysis
        if step_level == 'dimred':
            plotting.scatter(root=rootname, data_in=step_out, labels=y, true_labels=True)
            plotting.silhouette(root=rootname, labels=y, data_in=step_out, model=mdl_obj)

            if hasattr(mdl_obj, 'explained_variance_ratio_'):
                plotting.pcmagnitude(root=rootname,
                                     points=mdl_obj.explained_variance_ratio_,
                                     title='Explained variance ratio')
            if hasattr(mdl_obj, 'lambdas_'):
                plotting.pcmagnitude(root=rootname,
                                     points=mdl_obj.lambdas_/np.sum(mdl_obj.lambdas_),
                                     title='Normalized eigenvalues of the centered'
                                           ' kernel matrix')
        if step_level == 'clustering':
            if hasattr(mdl_obj, 'affinity_matrix_'):
                try:
                    n_clusters = mdl_obj.__dict__.get('cluster_centers_',
                                                      np.empty(0)).shape[0]
                except:
                    n_clusters = 0
                if hasattr(mdl_obj, 'n_clusters'):
                    n_clusters = mdl_obj.n_clusters

                plotting.eigs(root=rootname, affinity=mdl_obj.affinity_matrix_,
                              n_clusters=n_clusters,
                              title='Eigenvalues of the graph associated to '
                                    'the affinity matrix')
            if hasattr(mdl_obj, 'cluster_centers_'):
                _est_name = mdl_obj.__dict__.get('estimator_name', '') or \
                    type(mdl_obj).__name__
                if _est_name != 'AffinityPropagation':
                    # disable the voronoi plot for affinity prop
                    plotting.voronoi(root=rootname, labels=y, data_in=step_in,
                                     model=voronoi_mdl_obj)
            elif hasattr(mdl_obj, 'n_leaves_'):
                plotting.tree(root=rootname, data_in=step_in,
                              labels=y, index=index, model=mdl_obj)
                plotting.dendrogram(root=rootname, data_in=step_in,
                                    labels=y, index=index, model=mdl_obj)

            plotting.scatter(root=rootname, labels=step_out,
                             data_in=step_in, model=mdl_obj)
            plotting.silhouette(root=rootname, labels=step_out,
                                data_in=step_in, model=mdl_obj)
            est_clst_perf(root=rootname, data_in=step_in, labels=step_out,
                          t_labels=y, model=mdl_obj, metric=metric)


@timed
def analyze(input_dict, root, y=None, feat_names=None, index=None, **kwargs):
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

    index : list of integers (or strings)
        This is the samples identifier, if provided as first column (or row) of
        of the input file. Otherwise it is just an incremental range of size
        n_samples.

    kwargs : dictionary
        Additional optional parameters. In particular it can contain
        'plotting_context' and 'file_format' variables, if specified in
        the config file.
    """
    if GLOBAL_INFO:
        logging.info(GLOBAL_INFO)
    if kwargs.get('plotting_context', None):
        sns.set_context(kwargs.get('plotting_context'))

    file_formats = ('png', 'pdf')
    ff = kwargs.get('file_format', file_formats[0]).lower()

    if ff not in file_formats:
        logging.warning("File format unknown. "
                        "Please select one of %s", file_formats)
        plotting.DEFAULT_EXT = file_formats[0]
    else:
        plotting.DEFAULT_EXT = ff
    logging.info("File format set to %s", plotting.DEFAULT_EXT)
    lock = mp.Lock()
    ps = []
    for elem in items_iterator(input_dict):
        p = mp.Process(target=analysis_worker,
                       args=(elem, root, y, feat_names, index, lock))
        p.start()
        ps.append(p)

    for p in ps:
        p.join()

    # Create summary_scores.{txt, tex}
    make_df_clst_perf(root)

    # Compile tex
    try:
        with open(os.devnull, 'w') as devnull:
            # Someone may not have pdflatex installed
            subprocess.call(["pdflatex",
                             os.path.join(root, "summary_scores.tex")],
                            stdout=devnull, stderr=devnull)
            logging.info("PDF compilation done.")
        shutil.move("summary_scores.pdf",
                    os.path.join(root, "summary_scores.pdf"))
        os.remove("summary_scores.aux")
        os.remove("summary_scores.log")
        logging.info(".aux and .log cleaned")
    except StandardError:
        from sys import platform
        logging.warning("Suitable pdflatex installation not found.")
        if platform not in ["linux", "linux2", "darwin"]:
            logging.warning("Your operating system may not support"
                            "summary_scores.tex automatic pdf compilation.")
