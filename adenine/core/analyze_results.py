#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def make_scatter(root = (), embedding = (), model_param = (), trueLabel = np.nan):
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
    
    # Define plot color
    if not np.isnan(trueLabel[0]):
        y = trueLabel # use the labels if provided
        _hue = 'Classes'
    else:
        y = np.zeros((n_samples))
        _hue = ' '
    
    # Define the fileName
    fileName = os.path.basename(root)
    # Define the plot title
    for i, t in enumerate(root.split(os.sep)): # something like ['results', 'ade_debug_', 'Standardize', 'PCA']
        if t[0:5] == '_ade': break
    title = str("$\mapsto$").join(root.split(os.sep)[i-1:])
    
    # Create pandas data frame (needed by sns)
    X = embedding[:,:2]
    df = pd.DataFrame(data = np.hstack((X,y[:,np.newaxis])), columns = ["$x_1$","$x_2$",_hue])
    # Generate seaborn plot
    g = sns.FacetGrid(df, hue=_hue, palette="Set1", size=5, legend_out=False)
    g.map(plt.scatter, "$x_1$", "$x_2$", s=100, linewidth=.5, edgecolor="white")
    if _hue != ' ': g.add_legend() #!! customize legend
    # g.set_xticklabels([])
    # g.set_yticklabels([])
    plt.title(title)
    plt.savefig(os.path.join(root,fileName))
    
def make_voronoi(root = (), data_in = (), model_param = (), trueLabel = np.nan, labels = ()):
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
    """
    n_samples, n_dim = data.shape
    
    # Define plot color
    if not np.isnan(trueLabel[0]):
        y = trueLabel # use the labels if provided
        _hue = 'Classes'
    else:
        y = np.zeros((n_samples))
        _hue = ' '
    
    # --- TEMP: reduce the dimensionality by PCA just in order to plot the background of the voronoi tessellation
    if n_dim > 2:
        embedding = PCA(n_components=2).fit_transform(data)
        kmeans = KMeans(init='k-means++', n_clusters=n_digits)
        kmeans.fit(embedding)
    else:
        embedding = data
    # --- END TEMP
    
    # Seaborn Plot
    title = 'Voronoi Tessellation'
    X = embedding[:,:2]
    df = pd.DataFrame(data = np.hstack((X,y[:,np.newaxis])), columns = ["$x_1$","$x_2$",_hue])
    # Generate seaborn plot
    g = sns.FacetGrid(df, hue=_hue, palette="Set1", size=5, legend_out=False)
    g.map(plt.scatter, "$x_1$", "$x_2$", s=100, linewidth=.5, edgecolor="white")
    if _hue != ' ': g.add_legend() #!! customize legend
    plt.title(title)
    # #--------------------------#
    # plt.scatter(X[:,0], X[:,1], s=50, c=y, alpha=0.5) 
    
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].
    # h = 0.001
    
    # Plot the decision boundary. For that, we will assign a color to each
    # x_min, x_max = reduced_data[:, 0].min() + 1, reduced_data[:, 0].max() - 1
    # y_min, y_max = reduced_data[:, 1].min() + 1, reduced_data[:, 1].max() - 1
    x_min, x_max = reduced_data[:, 0].min(), reduced_data[:, 0].max()
    y_min, y_max = reduced_data[:, 1].min(), reduced_data[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.imshow(Z, interpolation='nearest',
               # extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               # extent=(-4, 4, -4, 4),
               extent = (np.round(x_min-1), np.round(x_max+1), np.round(y_min-1), np.round(y_max+1)),
               cmap=plt.get_cmap('Pastel1'), aspect = 'auto')
               # cmap=colmap,
               # aspect='auto', origin='lower')
    
    centroids = kmeans.cluster_centers_
    print centroids

    plt.show()
    
    

def est_clst_perf(root = (), label = (), trueLabel = np.nan, model_param = ()):
    """Estimate the clustering performance.
    
    This function estimate the clustering performance by means of several indexes. Then eventually saves the results in a tree-like structure in the root folder.
    
    Parameters
    -----------
    root : string
        The root path for the output creation
        
    label : array of float, shape : n_samples
        The label assignment performed by the clusterin algorithm.
        
    trueLabel : array of float, shape : n_samples
        The true label vector; np.nan if missing.
    
    model_param : dictionary
        The parameters of the clustering algorithm.
    """
    pass


def get_step_attributes(step = (), pos = ()):
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
    """
    name = step[0]
    level = step[1] # {imputing, preproc, dimred, clustering}
    param = step[2]
    data_out = step[3]
    data_in = step[4]
    if level.lower() == 'none' and pos == 0: level = 'preproc'
    if level.lower() == 'none' and pos == 1: level = 'dimred'
    
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
            
    logging.info("{} : {}".format(level,name)) 
    return name, level, param, data_out, data_in

def start(inputDict = (), rootFolder = (), y = np.nan, feat_names = (), class_names = ()):
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
        outFolder = '' # where the results will be placed
        logging.info("------\n{} : \n".format(pipe))
        for i, step in enumerate(sorted(inputDict[pipe].keys())):
            
            # Tree-like folder structure definition
            step_name, step_level, step_param, step_out, step_in = get_step_attributes(inputDict[pipe][step], pos = i)
            
            # Output folder definition & creation
            outFolder = os.path.join(outFolder,step_name)
            if not os.path.exists(os.path.join(rootFolder, outFolder)):
                os.makedirs(os.path.join(rootFolder, outFolder))
            
            # Launch analysis
            if step_level == 'dimred':
                make_scatter(root = os.path.join(rootFolder, outFolder), embedding = step_out, trueLabel = y, model_param = step_param, class_names = class_names)
            if step_level == 'clustering':
                make_voronoi(root = os.path.join(rootFolder, outFolder), label = step_out, trueLabel = y, model_param = step_param, data_in = step_in)
                # est_clst_perf(root = os.path.join(rootFolder, outFolder), label = step_res, trueLabel = y, model_param = step_param)
            
            
            
            