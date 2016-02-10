#!/usr/bin/python -W ignore::DeprecationWarning
# -*- coding: utf-8 -*-

import os
import logging
import cPickle as pkl
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.cluster.hierarchy import linkage as sp_linkage

def make_scatter(root=(), embedding=(), model_param=(), trueLabel=np.nan):
    """Generate and save the scatter plot of the dimensionality reduced data set.

    This function generates the scatter plot representing the dimensionality reduced data set. The plots will be saved into the root folder in a tree-like structure.

    Parameters
    -----------
    root : string
        The root path for the output creation

    embedding : array of float, shape : (n_samples, n_dimensions)
        The low space embedding estimated by the dimensinality reduction and manifold learning algorithm.

    model_param : dictionary
        The parameters of the dimensionality reduciont and manifold learning algorithm.

    trueLabel : array of float, shape : n_samples
        The true label vector; np.nan if missing (useful for plotting reasons).
    """
    n_samples, n_dim = embedding.shape

    # print trueLabel
    # raise ValueError('DEBUG')

    # Define plot color
#    if not np.isnan(trueLabel[0]):
    if trueLabel is None or trueLabel[0] == np.nan:
        y = np.zeros((n_samples))
        _hue = ' '
    else:
        y = trueLabel # use the labels if provided
        _hue = 'Classes'

    # Define the fileName
    fileName = os.path.basename(root)
    # Define the plot title
    for i, t in enumerate(root.split(os.sep)): # something like ['results', 'ade_debug_', 'Standardize', 'PCA']
        if t[0:5] == '_ade': break
    title = str("$\mapsto$").join(root.split(os.sep)[i-1:])

    # Create pandas data frame (needed by sns)
    X = embedding[:,:2]
    df = pd.DataFrame(data=np.hstack((X,y[:,np.newaxis])), columns=["$x_1$","$x_2$",_hue])
    # Generate seaborn plot
    g = sns.FacetGrid(df, hue=_hue, palette="Set1", size=5, legend_out=False)
    g.map(plt.scatter, "$x_1$", "$x_2$", s=100, linewidth=.5, edgecolor="white")
    if _hue != ' ': g.add_legend() #!! customize legend
    # g.set_xticklabels([])
    # g.set_yticklabels([])
    plt.title(title)
    plt.savefig(os.path.join(root,fileName))
    logging.info('Figured saved {}'.format(os.path.join(root,fileName)))
    plt.close()

def make_voronoi(root=(), data_in=(), model_param=(), trueLabel=np.nan, labels=(), model=()):
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

    trueLabel : array of float, shape : n_samples
        The true label vector; np.nan if missing (useful for plotting reasons).

    labels : array of int, shape : n_samples
        The result of the clustering step.

    model : sklearn or sklearn-like object
        An instance of the class that evaluates a step. In particular this must be a clustering model provided with the clusters_centers_ attribute (e.g. KMeans).
    """
    n_samples, n_dim = data_in.shape

    # Define plot color
    #if not np.isnan(trueLabel[0]):

    if trueLabel is None or trueLabel == np.nan:
        y = np.zeros((n_samples))
        _hue = ' '
    else:
        y = trueLabel # use the labels if provided
        _hue = 'Classes'

    # Define the fileName
    fileName = os.path.basename(root)
    # Define the plot title
    for i, t in enumerate(root.split(os.sep)): # something like ['results', 'ade_debug_', 'Standardize', 'PCA']
        if t[0:5] == '_ade': break
    title = str("$\mapsto$").join(root.split(os.sep)[i-2:])

    # Seaborn scatter Plot
    X = data_in[:,:2]
    df = pd.DataFrame(data=np.hstack((X,y[:,np.newaxis])), columns=["$x_1$","$x_2$",_hue])
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

    fileName = os.path.basename(root)
    plt.savefig(os.path.join(root,fileName))
    logging.info('Figured saved {}'.format(os.path.join(root,fileName)))


def est_clst_perf(root=(), data_in=(), label=(), trueLabel=np.nan, model=(), metric='euclidean'):
    """Estimate the clustering performance.

    This function estimate the clustering performance by means of several indexes. Then eventually saves the results in a tree-like structure in the root folder.

    Parameters
    -----------
    root : string
        The root path for the output creation

    data_in : array of float, shape : (n_samples, n_dimensions)
        The low space embedding estimated by the dimensinality reduction and manifold learning algorithm.

    label : array of float, shape : n_samples
        The label assignment performed by the clusterin algorithm.

    trueLabel : array of float, shape : n_samples
        The true label vector; np.nan if missing.

    model : sklearn or sklearn-like object
        An instance of the class that evaluates a step. In particular this must be a clustering model provided with the clusters_centers_ attribute (e.g. KMeans).
    """
    perf_out = dict()

    try:
        perf_out['silhouette'] = metrics.silhouette_score(data_in, label, metric=metric)

        if hasattr(model, 'inertia_'): # Sum of distances of samples to their closest cluster center.
            perf_out['inertia'] = mdl_obj.inertia_

        #if not np.isnan(np.array([trueLabel]).any()): # the next indexes need a gold standard
        # if not np.array([trueLabel]).any() == np.nan: # the next indexes need a gold standard
        if trueLabel is not None and not trueLabel == np.nan: # the next indexes need a gold standard
            perf_out['ari'] = metrics.adjusted_rand_score(trueLabel, label)
            perf_out['ami'] = metrics.adjusted_mutual_info_score(trueLabel, label)
            perf_out['homogeneity'] = metrics.homogeneity_score(trueLabel, label)
            perf_out['completeness'] = metrics.completeness_score(trueLabel, label)
            perf_out['v_measure'] = metrics.v_measure_score(trueLabel, label)

    except ValueError as e:
        logging.info("Clustering performance evaluation failed for {}".format(model))
        perf_out = {'empty': 0.0}

    # Define the fileName
    fileName = os.path.basename(root)
    with open(os.path.join(root,fileName+".txt"), "w") as f:
        f.write("------------------------------------\n")
        f.write("Adenine: Clustering Performance\n")
        f.write("------------------------------------\n")
        f.write("Index Name{}|{}Index Score\n".format(' '*10,' '*4))
        f.write("------------------------------------\n")
        for elem in sorted(perf_out.keys()):
            f.write("{}{}|{}{:.3}\n".format(elem,' '*(20-len(elem)),' '*4,perf_out[elem]))
            f.write("------------------------------------\n")

    # pkl Dump
    with open(os.path.join(root,fileName+'.pkl'), 'w+') as f:
        pkl.dump(perf_out, f)
    logging.info("Dumped : {}".format(os.path.join(root,fileName+'.pkl')))


def get_step_attributes(step=(), pos=()):
    """Get the attributes of the input step.

    This function returns the attributes (i.e. level, name, outcome) of the input step. This comes handy when dealing with steps with more than one parameter (e.g. KernelPCA 'poly' or 'rbf').

    Parameters
    -----------
    step : list
        A step coded by ade_run.py as [name, level, results, parameters]

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
        if param['affinity'] == 'precomputed':
            metric = 'precomputed'

    logging.info("{} : {}".format(level,name))
    return name, level, param, data_out, data_in, mdl_obj, voronoi_mdl_obj, metric

def make_tree(root=(), data_in=(), model_param=(), trueLabel=np.nan, labels=(), model=()):
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
    import itertools
    import pydot

    MAX_NODES = 150

    graph = pydot.Dot(graph_type='graph')

    ii = itertools.count(data_in.shape[0])
    for k, x in enumerate(model.children_):
        root_node = str(next(ii))
        left_edge = pydot.Edge(root_node, x[0])
        right_edge = pydot.Edge(root_node, x[1])
        graph.add_edge(right_edge)
        graph.add_edge(left_edge)
        if k > MAX_NODES:
            break

    # Define the filename
    filename = os.path.join(root, os.path.basename(root)+'_tree.png')
    graph.write_png(filename)
    logging.info('Figured saved {}'.format(filename))

def make_dendrogram(root=(), data_in=(), model_param=(), trueLabel=np.nan, labels=(), model=()):
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
        make_dendrograms = False
        if make_dendrograms:
            sns.set(font="monospace")
            for method in ['single','complete','average','weighted','centroid','median','ward']:
                print("Compute linkage matrix with metric={} ...".format(method))
                Z = sp_linkage(tmp, method=method, metric='euclidean')
                g = sns.clustermap(df.corr(), method=model.linkage, row_linkage=Z, col_linkage=Z)
                filename = os.path.join(root, os.path.basename(root)+'_'+method+'_dendrogram.png')
                g.savefig(filename)
                logging.info('Figured saved {}'.format(filename))
        else:
            import sys
            sys.path.insert(0, '/home/fede/Dropbox/projects/ig_network')
            from silhouette_score_plot import plot_avg_silhouette as main
            main(1. - tmp)
        return
    else:
        g = sns.clustermap(df.corr(), method=model.linkage, metric=model.affinity)

    # Define the fileName
    filename = os.path.join(root, os.path.basename(root)+'_dendrogram.png')
    g.savefig(filename)
    logging.info('Figured saved {}'.format(filename))


def make_scatterplot(root=(), data_in=(), model_param=(), trueLabel=np.nan, labels=(), model=(), n_dimensions=2):
    """Generate and save the scatter plot obtained from the clustering algorithm.

    This function generates the scatter plot obtained from the clustering algorithm applied on the data projected on a two-dimensional embedding. The color of the points in the plot is consistent with the label estimated by the algorithm. The plots will be saved into the appropriate folder of the tree-like structure created into the root folder.

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
    n_samples, n_dim = data_in.shape

    # Define plot color
    y = labels
    _hue = 'Estimated Labels'

    # Define the fileName
    fileName = os.path.basename(root)
    # Define the plot title
    for i, t in enumerate(root.split(os.sep)): # something like ['results', 'ade_debug_', 'Standardize', 'PCA']
        if t[0:5] == '_ade': break
    title = str("$\mapsto$").join(root.split(os.sep)[i-2:])

    # Seaborn scatter Plot
    X = data_in[:,:n_dimensions]
    if n_dimensions == 2:
        df = pd.DataFrame(data=np.hstack((X,y[:,np.newaxis])), columns=["$x_1$","$x_2$",_hue])
        # Generate seaborn plot
        g = sns.FacetGrid(df, hue=_hue, palette="Set1", size=5, legend_out=False)
        g.map(plt.scatter, "$x_1$", "$x_2$", s=100, linewidth=.5, edgecolor="white")
        if _hue != ' ': g.add_legend() #!! customize legend
    elif n_dimensions == 3:
        from mpl_toolkits.mplot3d import Axes3D
        f = plt.figure().gca(projection='3d')
        f.scatter(X[:,0],X[:,1],X[:,2],y,c=y,cmap='hot')

    plt.title(title)

    fileName = os.path.basename(root)
    plt.savefig(os.path.join(root,fileName+"_scatter"))
    logging.info('Figured saved {}'.format(os.path.join(root,fileName+"_scatter")))


def start(inputDict=(), rootFolder=(), y=np.nan, feat_names=(), class_names=()):
    """Analyze the results of ade_run.

    This function analyze the dictionary generated by ade_run, generates the plots, and saves them in a tree-like folder structure in rootFolder.

    Parameters
    -----------
    inputDict : dictionary
        The dictionary created by ade_run.py on some data.

    rootFolder : string
        The root path for the output creation

    y : array of float, shape : n_samples
        The label vector; np.nan if missing.

    feature_names : array of integers (or strings), shape : n_features
        The feature names; a range of numbers if missing.

    class_names : array of integers (or strings), shape : n_features
        The class names; a range of numbers if missing.
    """
    for pipe in inputDict: # iterating over a dictionary gives you the keys
        print("Analizing {} ...".format(pipe))
        outFolder = '' # where the results will be placed
        logging.info("------\n{} : \n".format(pipe))
        for i, step in enumerate(sorted(inputDict[pipe].keys())):

            # Tree-like folder structure definition
            step_name, step_level, step_param, step_out, step_in, mdl_obj, voronoi_mdl_obj, metric = get_step_attributes(inputDict[pipe][step], pos=i)


            # Output folder definition & creation
            outFolder = os.path.join(outFolder,step_name)
            if not os.path.exists(os.path.join(rootFolder, outFolder)):
                os.makedirs(os.path.join(rootFolder, outFolder))

            # Launch analysis
            if step_level == 'dimred':
                make_scatter(root=os.path.join(rootFolder, outFolder), embedding=step_out, trueLabel=y, model_param=step_param)
                plt.clf()
            if step_level == 'clustering':
                if hasattr(mdl_obj, 'cluster_centers_'):
                    make_voronoi(root=os.path.join(rootFolder, outFolder), labels=step_out, trueLabel=y, model_param=step_param, data_in=step_in, model=voronoi_mdl_obj)
                elif hasattr(mdl_obj, 'n_leaves_'):
                    make_tree(root=os.path.join(rootFolder, outFolder), labels=step_out, trueLabel=y, model_param=step_param, data_in=step_in, model=mdl_obj)

                    make_dendrogram(root=os.path.join(rootFolder, outFolder), labels=step_out, trueLabel=y, model_param=step_param, data_in=step_in, model=mdl_obj)

                make_scatterplot(root=os.path.join(rootFolder, outFolder), labels=step_out, trueLabel=y, model_param=step_param, data_in=step_in, model=mdl_obj)

                est_clst_perf(root=os.path.join(rootFolder, outFolder), data_in=step_in, label=step_out, trueLabel=y, metric=metric)
