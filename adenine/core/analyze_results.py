#!/usr/bin/python -W ignore::DeprecationWarning
# -*- coding: utf-8 -*-

import os, platform
# from joblib import Parallel, delayed
import logging
import cPickle as pkl
import numpy as np
import pandas as pd

# OSX matplotlib issues in multiprocessing
GLOBAL_INFO = ''  # to save info before logging is loaded
# if platform.system() == 'Darwin':
#     import matplotlib
#     backend = 'Qt4Agg'
#     matplotlib.use(backend)
#     GLOBAL_INFO = "matplotlib backend switched to {}".format(backend)

import matplotlib
matplotlib.use('AGG')

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from scipy.cluster.hierarchy import linkage as sp_linkage
import collections
import multiprocessing as mp

from adenine.utils.extra import next_color, reset_palette, title_from_filename

def make_scatter(root=(), data_in=(), model_param=(), labels=None, true_labels=False, model=()):
    """Generates and saves the scatter plot of the dimensionality reduced data set.

    This function generates the scatter plot representing the dimensionality reduced data set. The plots will be saved into the root folder in a tree-like structure.

    Parameters
    -----------
    root : string
        The root path for the output creation

    data_in : array of float, shape : (n_samples, n_dimensions)
        The low space embedding estimated by the dimensinality reduction and manifold learning algorithm.

    model_param : dictionary
        The parameters of the dimensionality reduciont and manifold learning algorithm.

    labels : array of float, shape : n_samples
        The label vector. It can contain true or estimated labels.

    true_labels : boolean
        Identify if labels contains true or estimated labels.

    model : sklearn or sklearn-like object
        An instance of the class that evaluates a step. In particular this must be a clustering model provided with the clusters_centers_ attribute (e.g. KMeans).
    """
    n_samples, n_dim = data_in.shape

    # Define plot color
    if labels is None:
        y = np.zeros((n_samples))
        _hue = ' '
    else:
        y = labels
        _hue = 'Classes' if true_labels else 'Estimated Labels'

    title = title_from_filename(root)

    # Seaborn scatter plot
    #2D plot
    X = data_in[:,:2]
    idx = np.argsort(y)
    df = pd.DataFrame(data=np.hstack((X[idx,:],y[idx,np.newaxis])), columns=["$x_1$","$x_2$",_hue])
    # Generate seaborn plot
    g = sns.FacetGrid(df, hue=_hue, palette="Set1", size=5, legend_out=False)
    g.map(plt.scatter, "$x_1$", "$x_2$", s=100, linewidth=.5, edgecolor="white")
    if _hue != ' ': g.add_legend() #!! customize legend
    # g.set_xticklabels([])
    # g.set_yticklabels([])
    plt.title(title)
    filename = os.path.join(root,os.path.basename(root)+"_scatter2D")
    plt.savefig(filename)
    logging.info('Figured saved {}'.format(filename))
    plt.close()

    #3D plot
    X = data_in[:,:3]
    if X.shape[1] < 3:
        logging.info(os.path.join(root,os.path.basename(root)+"_scatter3D") + ' cannot be generated (data have less than 3 dimensions)')
    else:
        try:
            from mpl_toolkits.mplot3d import Axes3D
            ax = plt.figure().gca(projection='3d')
            # ax.scatter(X[:,0], X[:,1], X[:,2], y, c=y, cmap='hot', s=100, linewidth=.5, edgecolor="white")
            d = collections.Counter(y)
            y = np.array(y)
            reset_palette()
            for colorid, k in enumerate(d):
                idx = np.where(y==k)[0]
                ax.plot(X[:,0][idx], X[:,1][idx], X[:,2][idx], 'o', c=next_color(), label=str(k), mew=.5, mec="white")

            ax.set_xlabel(r'$x_1$')
            ax.set_ylabel(r'$x_2$')
            ax.set_zlabel(r'$x_3$')
            ax.set_title(title)
            ax.legend(loc='upper left', numpoints=1, ncol=10, fontsize=8, bbox_to_anchor=(0, 0))
            # plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0111))
            filename = os.path.join(root,os.path.basename(root)+"_scatter3D")
            plt.savefig(filename)
            logging.info('Figured saved {}'.format(filename))
            plt.close()
        except Exception as e:
            logging.info('Error in 3D plot: ' + str(e))

    # seaborn pairplot
    n_cols = min(data_in.shape[1], 3)
    cols = ["$x_{}$".format(i+1) for i in range(n_cols)]
    X = data_in[:,:3]
    idx = np.argsort(y)
    df = pd.DataFrame(data=np.hstack((X[idx,:],y[idx,np.newaxis])), columns=cols+[_hue])
    g = sns.PairGrid(df, hue=_hue, palette="Set1", vars=cols)
    g = g.map_diag(plt.hist)#, palette="Set1")
    g = g.map_offdiag(plt.scatter, s=100, linewidth=.5, edgecolor="white")

    # g = sns.pairplot(df, hue=_hue, palette="Set1", vars=["$x_1$","$x_2$","$x_3$"]), size=5)
    if _hue != ' ': plt.legend(title=_hue,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize="large")
    plt.suptitle(title,x=0.6, y=1.01,fontsize="large")
    filename = os.path.join(root,os.path.basename(root)+"_pairgrid")
    g.savefig(filename)
    logging.info('Figured saved {}'.format(filename))
    plt.close()

def make_voronoi(root=(), data_in=(), model_param=(), labels=None, true_labels=False, model=()):
    """Generate and save the Voronoi tessellation obtained from the clustering algorithm.

    This function generates the Voronoi tessellation obtained from the clustering algorithm applied on the data projected on a two-dimensional embedding. The plots will be saved into the appropriate folder of the tree-like structure created into the root folder.

    Parameters
    -----------
    root : string
        The root path for the output creation

    data_in : array of float, shape : (n_samples, n_dimensions)
        The low space embedding estimated by the dimensinality reduction and manifold learning algorithm.

    model_param : dictionary
        The parameters of the dimensionality reduciont and manifold learning algorithm.

    labels : array of int, shape : n_samples
        The result of the clustering step.

    true_labels : boolean
        Identify if labels contains true or estimated labels.

    model : sklearn or sklearn-like object
        An instance of the class that evaluates a step. In particular this must be a clustering model provided with the clusters_centers_ attribute (e.g. KMeans).
    """
    n_samples, n_dim = data_in.shape

    # Define plot color
    if labels is None:
        y = np.zeros((n_samples))
        _hue = ' '
    else:
        y = labels # use the labels if provided
        _hue = 'Classes'

    title = title_from_filename(root)

    # Seaborn scatter Plot
    X = data_in[:,:2]
    idx = np.argsort(y)
    X = X[idx,:]
    y = y[idx,np.newaxis]
    df = pd.DataFrame(data=np.hstack((X, y)), columns=["$x_1$","$x_2$",_hue])
    # Generate seaborn plot
    g = sns.FacetGrid(df, hue=_hue, palette="Set1", size=5, legend_out=False)
    g.map(plt.scatter, "$x_1$", "$x_2$", s=100, linewidth=.5, edgecolor="white")
    if _hue != ' ': g.add_legend() #!! customize legend
    plt.title(title)

    # Add centroids
    if hasattr(model, 'cluster_centers_'):
        plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], s=100, marker='h', c='w')

    # Make and add to the Plot the decision boundary.
    npoints = 1000 # the number of points in that makes the background. Reducing this will decrease the quality of the voronoi background
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    offset = (x_max - x_min) / 5. + (y_max - y_min) / 5. # zoom out the plot a bit
    xx, yy = np.meshgrid(np.linspace(x_min-offset, x_max+offset, npoints), np.linspace(y_min-offset, y_max+offset, npoints))

    # Obtain labels for each point in mesh. Use last trained model.

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.get_cmap('Pastel1'), aspect='auto', origin='lower')

    plt.xlim([xx.min(), xx.max()])
    plt.ylim([yy.min(), yy.max()])

    filename = os.path.join(root,os.path.basename(root))
    plt.savefig(filename)
    logging.info('Figured saved {}'.format(filename))
    plt.close()


def est_clst_perf(root=(), data_in=(), labels=None, t_labels=None, model=(), metric='euclidean'):
    """Estimate the clustering performance.

    This function estimate the clustering performance by means of several indexes. Then eventually saves the results in a tree-like structure in the root folder.

    Parameters
    -----------
    root : string
        The root path for the output creation

    data_in : array of float, shape : (n_samples, n_dimensions)
        The low space embedding estimated by the dimensinality reduction and manifold learning algorithm.

    labels : array of float, shape : n_samples
        The label assignment performed by the clusterin algorithm.

    t_labels : array of float, shape : n_samples
        The true label vector; None if missing.

    model : sklearn or sklearn-like object
        An instance of the class that evaluates a step. In particular this must be a clustering model provided with the clusters_centers_ attribute (e.g. KMeans).
    """
    perf_out = dict()

    try:
        perf_out['silhouette'] = metrics.silhouette_score(data_in, labels, metric=metric)

        if hasattr(model, 'inertia_'):
            # Sum of distances of samples to their closest cluster center.
            perf_out['inertia'] = model.inertia_

        if t_labels is not None:
            # the next indexes need a gold standard
            perf_out['ari'] = metrics.adjusted_rand_score(t_labels, labels)
            perf_out['ami'] = metrics.adjusted_mutual_info_score(t_labels, labels)
            perf_out['homogeneity'] = metrics.homogeneity_score(t_labels, labels)
            perf_out['completeness'] = metrics.completeness_score(t_labels, labels)
            perf_out['v_measure'] = metrics.v_measure_score(t_labels, labels)

    except ValueError as e:
        logging.info("Clustering performance evaluation failed for {}".format(model))
        # perf_out = {'empty': 0.0}
        perf_out['###'] = 0.

    # Define the filename
    filename = os.path.join(root,os.path.basename(root))
    with open(filename+'_scores.txt', 'w') as f:
        f.write("------------------------------------\n")
        f.write("Adenine: Clustering Performance for \n")
        f.write("\n")
        f.write(title_from_filename(root, " --> ") + "\n")
        f.write("------------------------------------\n")
        f.write("Index Name{}|{}Index Score\n".format(' '*10,' '*4))
        f.write("------------------------------------\n")
        for elem in sorted(perf_out.keys()):
            f.write("{}{}|{}{:.3}\n".format(elem,' '*(20-len(elem)),' '*4,perf_out[elem]))
            f.write("------------------------------------\n")

    # pkl Dump
    with open(filename+'_scores.pkl', 'w+') as f:
        pkl.dump(perf_out, f)
    logging.info("Dumped : {}".format(filename+'_scores.pkl'))


def get_step_attributes(step=(), pos=()):
    """Get the attributes of the input step.

    This function returns the attributes (i.e. level, name, outcome) of the input step. This comes handy when dealing with steps with more than one parameter (e.g. KernelPCA 'poly' or 'rbf').

    Parameters
    -----------
    step : list
        A step coded by ade_run.py as [name, level, param, data_out, data_in, mdl obj, voronoi_mdl_obj]

    pos : int
        The position of the step inside the pipeline.

    Returns
    -----------
    name : string
        A unique name for the step (e.g. KernelPCA_rbf).

    level : {imputing, preproc, dimred, clustering}
        The step level.

    data_out : array of float, shape : (n_samples, n_out)
        Where n_out is n_dimensions for dimensionality reduction step, or 1 for clustering.

    data_in : array of float, shape : (n_samples, n_in)
        Where n_in is n_dimensions for preprocessing/imputing/dimensionality reduction step, or n_dim for clustering (because the data have already been dimensionality reduced).

    param : dictionary
        The parameters of the sklearn object implementing the algorithm.

    mdl_obj : sklearn or sklearn-like object
        This is an instance of the class that evaluates a step.
    """
    name = step[0]
    level = step[1] # {imputing, preproc, dimred, clustering}
    param = step[2]
    data_out = step[3]
    data_in = step[4]
    mdl_obj = step[5]
    voronoi_mdl_obj = step[6]
    if level.lower() == 'none' and pos == 0: level = 'preproc'
    if level.lower() == 'none' and pos == 1: level = 'dimred'
    metric = 'euclidean'

    # Append additional parameters in the step name
    if name == 'KernelPCA':
        name += '_'+param['kernel']
    elif name == 'LLE':
        name += '_'+param['method']
    elif name == 'MDS':
        if param['metric'] == 'True':
            name += '_metric'
        else:
            name += '_nonmetric'
    elif name == 'Hierarchical':
        name += '_'+param['affinity']+'_'+param['linkage']

    if param.get('affinity', '') == 'precomputed':
        metric = 'precomputed'

    n_clusters = param.get('n_clusters', 0) or param.get('best_estimator_', dict()).get('n_clusters', 0)
    if n_clusters > 0:
        name += '_' + str(n_clusters) + '-clusts'

    logging.info("{} : {}".format(level,name))
    return name, level, param, data_out, data_in, mdl_obj, voronoi_mdl_obj, metric

def make_tree(root=(), data_in=(), model_param=(), trueLabel=None, labels=(), model=()):
    """Generate and save the tree structure obtained from the clustering algorithm.

    This function generates the tree obtained from the clustering algorithm applied on the data. The plots will be saved into the appropriate folder of the tree-like structure created into the root folder.

    Parameters
    -----------
    root : string
        The root path for the output creation

    data_in : array of float, shape : (n_samples, n_dimensions)
        The low space embedding estimated by the dimensinality reduction and manifold learning algorithm.

    model_param : dictionary
        The parameters of the dimensionality reduciont and manifold learning algorithm.

    trueLabel : array of float, shape : n_samples
        The true label vector; np.nan if missing (useful for plotting reasons).

    labels : array of int, shape : n_samples
        The result of the clustering step.

    model : sklearn or sklearn-like object
        An instance of the class that evaluates a step. In particular this must be a clustering model provided with the clusters_centers_ attribute (e.g. KMeans).
    """
    filename = os.path.join(root, os.path.basename(root)+'_tree.pdf')
    try:
        import itertools
        import pydot

        graph = pydot.Dot(graph_type='graph')

        ii = itertools.count(data_in.shape[0])
        for k, x in enumerate(model.children_):
            root_node = next(ii)
            left_edge = pydot.Edge(root_node, x[0])
            right_edge = pydot.Edge(root_node, x[1])
            graph.add_edge(right_edge)
            graph.add_edge(left_edge)

        # graph.write_png(filename[:-2]+"ng")
        graph.write_pdf(filename)
        logging.info('Figured saved {}'.format(filename))
    except:
        logging.info('Cannot create {}'.format(filename))

def make_dendrogram(root=(), data_in=(), model_param=(), trueLabel=None, labels=(), model=()):
    """Generate and save the dendrogram obtained from the clustering algorithm.

    This function generates the dendrogram obtained from the clustering algorithm applied on the data. The plots will be saved into the appropriate folder of the tree-like structure created into the root folder.

    Parameters
    -----------
    root : string
        The root path for the output creation

    data_in : array of float, shape : (n_samples, n_dimensions)
        The low space embedding estimated by the dimensinality reduction and manifold learning algorithm.

    model_param : dictionary
        The parameters of the dimensionality reduciont and manifold learning algorithm.

    trueLabel : array of float, shape : n_samples
        The true label vector; np.nan if missing (useful for plotting reasons).

    labels : array of int, shape : n_samples
        The result of the clustering step.

    model : sklearn or sklearn-like object
        An instance of the class that evaluates a step. In particular this must be a clustering model provided with the clusters_centers_ attribute (e.g. KMeans).
    """
    # tmp = np.hstack((np.arange(0,data_in.shape[0],1)[:,np.newaxis], data_in[:,0][:,np.newaxis], data_in[:,1][:,np.newaxis]))
    tmp = data_in
    col = ["$x_{"+str(i)+"}$" for i in np.arange(0, data_in.shape[1], 1)]
    df = pd.DataFrame(data=tmp, columns=col)

    if model.affinity == 'precomputed': # TODO sistemare, fede
        # tmp is the distance matrix
        make_dendrograms = False
        if make_dendrograms:
            sns.set(font="monospace")
            for method in ['single','complete','average','weighted','centroid','median','ward']:
                # print("Compute linkage matrix with metric={} ...".format(method))
                Z = sp_linkage(tmp, method=method, metric='euclidean')
                g = sns.clustermap(df.corr(), method=method, row_linkage=Z, col_linkage=Z)
                filename = os.path.join(root, '_'.join((os.path.basename(root), method, '_dendrogram.png')))
                g.savefig(filename)
                logging.info('Figured saved {}'.format(filename))
                plt.close()
        avg_sil = True
        if avg_sil:
            try:
                from ignet.plotting.silhouette_hierarchical import plot_avg_silhouette
                filename = plot_avg_silhouette(tmp)
                logging.info('Figured saved {}'.format(filename))
            except:
                logging.warn("Cannot import name {}".format('ignet.plotting'))
        return

    # workaround to a different name used for manhatta / cityblock distance
    if model.affinity == 'manhattan': model.affinity = 'cityblock'

    g = sns.clustermap(df, method=model.linkage, metric=model.affinity, cmap='coolwarm')
    plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, fontsize=5)
    filename = os.path.join(root, os.path.basename(root)+'_dendrogram.png')
    g.savefig(filename)
    logging.info('Figured saved {}'.format(filename))
    plt.close()

def plot_PCmagnitude(root=(), points=(), title='', ylabel=''):
    """Generate and save the plot representing the trend of principal components magnitude.

    Parameters
    -----------

    rootFolder : string
        The root path for the output creation

    points : array of float, shape : n_components
        This could be the explained variance ratio or the eigenvalues of the centered matrix, according to the PCA algorithm of choice, respectively: PCA or KernelPCA.

    title : string
        Plot title
    """
    plt.plot(np.arange(1, len(points)+1), points, '-o')
    plt.title(title)
    plt.grid('on')
    plt.ylabel(ylabel)
    plt.xlim([1,min(20,len(points)+1)]) # Show maximum 20 components
    plt.ylim([0,1])
    filename = os.path.join(root,os.path.basename(root)+"_magnitude")
    plt.savefig(filename)
    plt.close()

def plot_eigs(root='', affinity=(), n_clusters=0, title='', ylabel='', normalised=True):
    """Generate and save the plot representing the eigenvalues of the Laplacian
    associated to data affinity matrix.

    Parameters
    -----------

    rootFolder : string
        The root path for the output creation

    points : array of float, shape : n_components
        This could be the explained variance ratio or the eigenvalues of the centered matrix, according to the PCA algorithm of choice, respectively: PCA or KernelPCA.

    title : string
        Plot title
    """
    W = affinity - np.diag(np.diag(affinity))
    D = np.diag([np.sum(x) for x in W])
    L = D - W
    if normalised:
        # aux = np.linalg.inv(np.diag([np.sqrt(np.sum(x)) for x in W]))
        aux =  np.diag(1. / np.array([np.sqrt(np.sum(x)) for x in W]))
        L = np.eye(L.shape[0]) - (np.dot(np.dot(aux,W),aux)) # normalised L

    w, v = np.linalg.eig(L)
    w = np.array(sorted(np.abs(w)))
    plt.plot(np.arange(1, len(w)+1), w, '-o')
    plt.title(title)
    plt.grid('on')
    plt.ylabel(ylabel)
    plt.xlim([1,min(20,len(w)+1)]) # Show maximum 20 components
    if n_clusters > 0:
        plt.axvline(x=n_clusters+.5, linestyle='--', color='r', label='selected clusters')
    plt.legend(loc='upper right', numpoints=1, ncol=10, fontsize=8)#, bbox_to_anchor=(1, 1))
    filename = os.path.join(root,os.path.basename(root)+"_eigenvals")
    plt.savefig(filename)
    plt.close()

def make_df_clst_perf(rootFolder):
    df = pd.DataFrame(columns=['pipeline','silhouette', 'inertia', 'ari', 'ami', 'homogeneity', 'completeness', 'v_measure'])
    for root, directories, filenames in os.walk(rootFolder):
        for fn in filenames:
            if fn.endswith('_scores.pkl'):
                with open(os.path.join(root, fn), 'r') as f:
                    perf_out = pkl.load(f)
                perf_out['pipeline'] = title_from_filename(root, step_sep=" --> ")
                df = df.append(perf_out, ignore_index=True)
    df = df.fillna('---')
    size_pipe = max([len(p) for p in df['pipeline']]+[len('preprocess --> dim red --> clustering')])
    with open(os.path.join(rootFolder,'summary_scores.txt'), 'w') as f:
        header = "preprocess --> dim red --> clustering{0}|{1}ami|{1}ari|{2}completeness|{2}homogeneity|{2}v_measure|{2}inertia|{2}silhouette\n".format(' '*(size_pipe-len("preprocess --> dim red --> clustering")),' '*4,' ')
        f.write("-"*len(header) + "\n")
        f.write("Adenine: Clustering Performance for each pipeline\n")
        f.write("-"*len(header) + "\n")
        f.write(header)
        f.write("-"*len(header) + "\n")
        for _ in df.iterrows():
            row = _[1]
            ami = '{:.3}'.format(row['ami'])
            ari = '{:.3}'.format(row['ari'])
            com = '{:.3}'.format(row['completeness'])
            hom = '{:.3}'.format(row['homogeneity'])
            vme = '{:.3}'.format(row['v_measure'])
            ine = '{:.3}'.format(row['inertia'])
            sil = '{:.3}'.format(row['silhouette'])
            # f.write("{}{}|{}{:.3f}|{}{:.3f}|{}{:.3f}|{}{:.3f}|{}{:.3f}|{}{:.3f}|{}{:.3f}\n"
            f.write("{}{}|{}{}|{}{}|{}{}|{}{}|{}{}|{}{}|{}{}\n"
            .format(row['pipeline'],' '*(size_pipe-len(row['pipeline'])),
            ' '*(len('    ami')-len(ami)), ami,
            ' '*(len('    ari')-len(ari)), ari,
            ' '*(len(' completeness')-len(com)), com,
            ' '*(len(' homogeneity')-len(hom)), hom,
            ' '*(len(' v_measure')-len(vme)), vme,
            ' '*(len(' inertia')-len(ine)), ine,
            ' '*(len(' silhouette')-len(sil)), sil
            ))
        f.write("-"*len(header) + "\n")
    # df.to_csv(os.path.join(rootFolder,'all_scores.csv'), na_rep='-', index_label=False, index=False)


def analysis_worker(elem, rootFolder, y, feat_names, class_names, lock):
    """Parallel pipelines analysis.

    Parameters
    -----------

    rootFolder : string
        The root path for the output creation

    y : array of float, shape : n_samples
        The label vector; None if missing.

    feature_names : array of integers (or strings), shape : n_features
        The feature names; a range of numbers if missing.

    class_names : array of integers (or strings), shape : n_features
        The class names; a range of numbers if missing.
    """
    # Getting pipeID and content
    pipe = elem[0]
    content = elem[1]

    outFolder = '' # where the results will be placed
    logging.info("------\n{} : \n".format(pipe))
    for i, step in enumerate(sorted(content.keys())):

        # Tree-like folder structure definition
        step_name, step_level, step_param, step_out, step_in, mdl_obj, voronoi_mdl_obj, metric = get_step_attributes(content[step], pos=i)

        # Output folder definition & creation
        outFolder = os.path.join(outFolder, step_name)
        rootname = os.path.join(rootFolder, outFolder)
        with lock:
            if not os.path.exists(rootname):
                os.makedirs(rootname)

        # Launch analysis
        if step_level == 'dimred':
            make_scatter(root=rootname, data_in=step_out, labels=y, true_labels=True, model_param=step_param)
            if hasattr(mdl_obj, 'explained_variance_ratio_'):
                plot_PCmagnitude(root=rootname, points=mdl_obj.explained_variance_ratio_, title='Explained variance ratio')
            if hasattr(mdl_obj, 'lambdas_'):
                plot_PCmagnitude(root=rootname, points=mdl_obj.lambdas_/np.sum(mdl_obj.lambdas_), title='Normalized eigenvalues of the centered kernel matrix')
        if step_level == 'clustering':
            if hasattr(mdl_obj, 'affinity_matrix_'):
                plot_eigs(root=rootname, affinity=mdl_obj.affinity_matrix_, n_clusters=mdl_obj.get_params()['n_clusters'], title='Eigenvalues of the graph associated to the affinity matrix')
            if hasattr(mdl_obj, 'cluster_centers_'):
                # make_voronoi(root=rootname, labels=step_out, trueLabel=y, model_param=step_param, data_in=step_in, model=voronoi_mdl_obj)
                make_voronoi(root=rootname, labels=y, true_labels=True, model_param=step_param, data_in=step_in, model=voronoi_mdl_obj)
            elif hasattr(mdl_obj, 'n_leaves_'):
                make_tree(root=rootname, labels=step_out, trueLabel=y, model_param=step_param, data_in=step_in, model=mdl_obj)

                make_dendrogram(root=rootname, labels=step_out, trueLabel=y, model_param=step_param, data_in=step_in, model=mdl_obj)

            # make_scatterplot(root=rootname, labels=step_out, model_param=step_param, data_in=step_in, model=mdl_obj)
            make_scatter(root=rootname, labels=step_out, model_param=step_param, data_in=step_in, model=mdl_obj)

            est_clst_perf(root=rootname, data_in=step_in, labels=step_out, t_labels=y, model=mdl_obj, metric=metric)

    make_df_clst_perf(rootFolder)


def start(inputDict=(), rootFolder=(), y=None, feat_names=(), class_names=()):
    """Analyze the results of ade_run.

    This function analyze the dictionary generated by ade_run, generates the plots, and saves them in a tree-like folder structure in rootFolder.

    Parameters
    -----------
    inputDict : dictionary
        The dictionary created by ade_run.py on some data.

    rootFolder : string
        The root path for the output creation

    y : array of float, shape : n_samples
        The label vector; None if missing.

    feature_names : array of integers (or strings), shape : n_features
        The feature names; a range of numbers if missing.

    class_names : array of integers (or strings), shape : n_features
        The class names; a range of numbers if missing.
    """
    if GLOBAL_INFO: logging.info(GLOBAL_INFO)
    lock = mp.Lock()
    # Parallel(n_jobs=len(inputDict))(delayed(analysis_worker)(elem,rootFolder,y,feat_names,class_names,lock) for elem in inputDict.iteritems())
    ps = []
    for elem in inputDict.iteritems():
        p = mp.Process(target=analysis_worker,
                       args=(elem,rootFolder,y,feat_names,class_names,lock))
        p.start()
        ps.append(p)

    for p in ps:
        p.join()
