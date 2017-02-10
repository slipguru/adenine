#!/usr/bin/env python
# -*- coding: utf-8 -*-

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Federico Tomasi, Annalisa Barla
#
# FreeBSD License
######################################################################

import logging
import warnings

import numpy as np

# from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import Imputer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score as sil
from sklearn.metrics.pairwise import pairwise_distances

from adenine.externals import AgglomerativeClustering
from sklearn.externals.joblib import Memory

# Legacy import
try:
    from sklearn.model_selection import GridSearchCV
except ImportError:
    from sklearn.grid_search import GridSearchCV


class DummyNone:
    """Dummy class that does nothing.

    It is a sklearn 'transforms', it implements both a fit and a transform
    method and it just returns the data in input. It has been created only for
    consistency with sklearn.
    """

    def __init__(self, **kwargs):
        if kwargs.get('n_components', 0) > 0:
            self.n_components = kwargs.get('n_components')

    def fit(self, X):
        return self

    def transform(self, X):
        return X

    def get_params(self):
        return dict()


class Imputer(Imputer):
    """Extension of the sklearn.preprocessing.Imputer class.

    This class adds the nearest_neighbors data imputing strategy.
    """

    def fit(self, X, y=None):
        if self.strategy.lower() in ['nearest_neighbors', 'nn']:
            self._nn_fit(X)
        else:
            if y is not None:
                super(Imputer, self).fit(X, y)
            else:
                super(Imputer, self).fit(X)
        return self

    def transform(self, X):
        if self.strategy.lower() in ['nearest_neighbors', 'nn']:
            # 1. Find missing values
            missing = self._get_mask(X, self.missing_values)
            # 2. Drop empty rows (I cannot deal with that)
            mask = ~np.prod(missing, axis=1, dtype=np.bool)
            X_copy = X[mask, :].copy()
            missing = missing[mask, :]

            # 3. Statistics init
            self.statistics_ = np.empty_like(X_copy)
            # self.statistics_ = np.zeros_like(X_copy)

            # 4. For each row that presents a True value in missing:
            #    drop the True column and get the first K Nearest Neighbors
            _cond = True
            count = 0
            while _cond and count < 100:
                for i, row in enumerate(missing):
                    if row.any():  # i.e. if True in row:
                        self._filling_worker(X_copy, row, i)
                _cond = np.isnan(self.statistics_).any()
                X_copy[missing] = self.statistics_[missing]
                count += 1

            # Log the failure
            if _cond:
                logging.info("Data imputing partially failed.")
                print("Data imputing partially failed.")

            # X_copy[missing] = self.statistics_[missing]
            return X_copy
        else:
            return super(Imputer, self).transform(X)

    def _get_mask(self, X, value_to_mask):
        """Compute the boolean mask X == missing_values.

        [copy/pasted from sklearn.preprocessing]
        """
        if value_to_mask == "NaN" or np.isnan(value_to_mask):
            return np.isnan(X)
        else:
            return X == value_to_mask

    def _get_row_indexes(self, c_idx):
        """
        Get which samples do not have missing values or have the same missing
        value as the current sample.
        """
        # Drop the column with missing values in the i-th sample
        _missing = self.X_missing.copy()
        _missing = _missing[:, c_idx]

        # Get the filled columns
        r_idx = []
        for k, r in enumerate(~_missing):
            # _not_row = [not j for j in r]
            if np.prod(r, dtype=np.bool):
                # it's like False is not in _not_row
                r_idx.append(k)

        return np.array(r_idx)

    def _filling_worker(self, X, row, i):
        """Worker for parallel execution of self._nn_fit()."""
        # the list of non-missing values for the i-th row
        c_idx = np.where(~row)[0]
        # c_idx = np.where([not j for j in row])[0]

        # Generate the training matrix (only the non-empty columns)
        r_idx = self._get_row_indexes(c_idx)
        # get the full matrix of possible neighbors
        Xtr = self.X_[r_idx[:, np.newaxis], c_idx]

        neigh = NearestNeighbors(n_neighbors=min(6, Xtr.shape[0]), n_jobs=1)

        # Get the nearest Neighbors
        neigh.fit(Xtr)

        with warnings.catch_warnings():  # shut-up deprecation warnings
            warnings.simplefilter("ignore")
            _nn_idx = neigh.kneighbors(X[i, c_idx], return_distance=False)[0]

        # Evaluate the average of the nearest Neighbors
        # build nearest neighbor matrix, skip the first one which is the same
        neighbors = self.X_[r_idx[_nn_idx], :]

        with warnings.catch_warnings():  # shut-up deprecation warnings
            warnings.simplefilter("ignore")
            _nanmean = np.nanmean(neighbors[:, np.where(row)[0]], axis=0)

        self.statistics_[i, row] = _nanmean

    def _nn_fit(self, X):
        """Impute the input data matrix using the Nearest Neighbors approach.

        This implementation follows, approximately, the strategy proposed in:
        [Hastie, Trevor, et al. "Imputing missing data for gene expression
        arrays." (1999): 1-7.]
        [Troyanskaya, Olga, et al. "Missing value estimation methods for DNA
        microarrays." Bioinformatics 17.6 (2001): 520-525.]
        """
        # 1. Find missing values
        self.X_missing = self._get_mask(X, self.missing_values)
        # 2. Drop empty rows (I cannot deal with that)
        self.X_mask = ~np.prod(self.X_missing, axis=1, dtype=np.bool)
        # ~ operator is doing this:
        self.X_ = X[self.X_mask, :].copy()

        # Preserve dimension of training input data in terms of empty rows
        self.X_missing = self.X_missing[self.X_mask, :]

        # 3. Statistics init
        self.statistics_ = np.empty_like(self.X_)

        # 4. For each row that presents a True value in missing:
        #    drop the True column and get the first K Nearest Neighbors
        _cond = True
        count = 0
        while _cond and count < 100:
            for i, row in enumerate(self.X_missing):
                if row.any():  # i.e. if True in row:
                    self._filling_worker(self.X_, row, i)
            _cond = np.isnan(self.statistics_).any()
            self.X_[self.X_missing] = self.statistics_[self.X_missing]
            count += 1

        # Log the failure
        if _cond:
            logging.info("Data imputing partially failed.")

        return self


class GridSearchCV(GridSearchCV):
    """Wrapper for sklearn's GridSearchCV.

    Automatically detects the optimal number of clusters for centroid-based
    algorithms like KMeans and Affinity Propagation.
    """
    def __init__(self, estimator, param_grid, scoring=None, fit_params=None,
                 n_jobs=1, iid=True, refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score='raise',
                 affinity='euclidean'):
        super(GridSearchCV, self).__init__(estimator, param_grid, scoring,
                                           fit_params, n_jobs, iid, refit,
                                           cv, verbose, pre_dispatch,
                                           error_score)
        self.affinity = affinity  # add the attribute affinity
        self.cluster_centers_ = None
        self.inertia_ = None
        self.n_clusters = None
        self.estimator_name = type(self.estimator).__name__

    def _sqrtn_heuristic(self, _n):
        """Heuristic for KMeans.

        n_clusters grid for KMeans: logaritmic scale in
        [2,...,log10(sqrt(n))] with max length = 30.
        For data-poor cases (i.e., when the log scale has fewer elements than
        the number of samples in each split), a linear scale is returned.
        """
        # The number of labelsmust be in 2 to n_samples - 1 (inclusive)
        n = _n // self.cv
        krange = np.unique(map(int, np.logspace(np.log10(2),
                                                np.log10(np.sqrt(n)), 30)))
        krange = krange[np.multiply(krange >= 2, krange <= n - 1)]

        # Data poor
        if len(krange) < n:
            krange = np.arange(2, n)
        return krange

    def _min_max_dist_heuristic(self, X, affinity):
        """Heuristic for AffinityPropagation.

        Preference grid for Affinity Propagation: linear scale in
        [min(similarity matrix),...,median(similarity matrix)]
        """
        S = -pairwise_distances(X, metric=affinity, squared=True)
        return np.unique(map(int, np.linspace(np.min(S), np.median(S), 30)))

    def fit(self, X, y=None):
        """Re-definition of the fit method.

        This new definition of the fit method sets the grid following a
        different heuristic according to the clustering algorithm
        """
        # Correct the number of splits for data-poor cases
        if X.shape[0] / np.float(self.cv) < 5:
            logging.info("[GridSearchCV] Data Poor: the number of splits {} "
                         "will be reduced to half the original "
                         "value".format(self.cv))
            self.cv //= 2

        # Pick the heuristic
        if type(self.estimator).__name__ == 'KMeans':
            # pick heuristic 1
            self.param_grid = {'n_clusters': self._sqrtn_heuristic(X.shape[0])}
        elif type(self.estimator).__name__ == 'AffinityPropagation':
            # pick heuristic 2
            self.param_grid = {'preference':
                               self._min_max_dist_heuristic(X, self.affinity)}

        # Then perform standard fit
        if y:
            super(GridSearchCV, self).fit(X, y)
        else:
            super(GridSearchCV, self).fit(X)

        # Propagate the cluster_centers_ attribute (needed for voronoi plot)
        if hasattr(self.best_estimator_, 'cluster_centers_'):
            # added for consistency only
            self.cluster_centers_ = self.best_estimator_.cluster_centers_

        # Propagate the inertia_ attribute
        if hasattr(self.best_estimator_, 'inertia_'):
            self.inertia_ = self.best_estimator_.inertia_

        # Propagate the n_clusters
        if hasattr(self.best_estimator_, 'cluster_centers_'):
            self.n_clusters = self.best_estimator_.cluster_centers_.shape[0]

        if hasattr(self.best_estimator_, 'affinity_matrix_'):
            self.affinity_matrix_ = self.best_estimator_.affinity_matrix_

        return self

    def get_params(self, deep=True):
        params_ = super(GridSearchCV, self).get_params(deep)
        params_['n_clusters'] = self.n_clusters
        params_['inertia_'] = self.inertia_
        params_['estimator_name'] = self.estimator_name
        return params_


def silhouette_score(estimator, X, y=None):
    """Scorer wrapper for metrics.silhouette_score."""
    if hasattr(estimator, 'affinity') and estimator.affinity == 'precomputed':
        return np.nan

    _y = estimator.predict(X)
    n_labels = len(np.unique(_y))
    if 1 < n_labels < X.shape[0]:
        return sil(X, _y)

    logging.warn("adenine.utils.extension.silhouette_score() returned NaN "
                 "because the number of labels is {}. Valid values are 2 "
                 "to n_samples - 1 (inclusive) = {}"
                 .format(n_labels, X.shape[0]-1))
    return np.nan


class KernelPCA(KernelPCA):
    """Extension of sklearn Kernel PCA.

    This KernelPCA class uses a different heuristic (w.r.t. sklearn's one) for
    the default value of gamma of rbf kernels.

    The default value of gamma is not 1 / n_features (as in sklearn), but it
    becomes 1 / (2 * sigma^2) where sigma is the output of
    self._autosigma(data, n_nearest_neighbor).
    """

    def __init__(self, n_components=None, kernel="linear",
                 gamma=None, degree=3, coef0=1, kernel_params=None,
                 alpha=1.0, fit_inverse_transform=False, eigen_solver='auto',
                 tol=0, max_iter=None, remove_zero_eig=False):
        super(KernelPCA, self).__init__(n_components, kernel,
                                        gamma, degree, coef0,
                                        kernel_params, alpha,
                                        fit_inverse_transform,
                                        eigen_solver, tol,
                                        max_iter, remove_zero_eig)

    def _autosigma(self, data, n_neighbors=5):
        """Compute the average n_neighbors distance of n p-dimensional points.

        Parameters
        -----------
        data : (n, p) data matrix
            The input data.

        n_neighbors : int
            The number of considered nearest neighbors (optional, default = 5).
        """
        # evaluate pairwise euclidean distances
        from sklearn.metrics.pairwise import pairwise_distances
        D = pairwise_distances(data, data)
        D.sort(axis=0)
        return np.mean(D[n_neighbors+1, :].ravel())

    def fit(self, X, **kwargs):
        # Apply the _autosigma heuristic
        if self.kernel == 'rbf' and self.gamma is None:
            self.gamma = 1.0 / (2 * self._autosigma(data=X)**2)
            # print("Gamma is: {}".format(self.gamma))
        super(KernelPCA, self).fit(X, **kwargs)


class AgglomerativeClustering(AgglomerativeClustering):
    """Extension of sklearn Agglomerative Clustering.

    This Agglomerative Clustering class, if required, can perform automatic
    discovery of the number of clusters.
    """

    def __init__(self, n_clusters=2, affinity="euclidean",
                 memory=Memory(cachedir=None, verbose=0),
                 connectivity=None, n_components=None,
                 compute_full_tree='auto', linkage='ward',
                 pooling_func=np.mean, return_distance=False):
        """Agglomerative Clustering.

        Recursively merges the pair of clusters that minimally increases
        a given linkage distance.

        Read more in the :ref:`User Guide <hierarchical_clustering>`.

        Parameters
        ----------
        n_clusters : int, default=2
            The number of clusters to find.

        connectivity : array-like or callable, optional
            Connectivity matrix. Defines for each sample the neighboring
            samples following a given structure of the data.
            This can be a connectivity matrix itself or a callable that
            transforms the data into a connectivity matrix, such as derived
            from kneighbors_graph. Default is None, i.e, the
            hierarchical clustering algorithm is unstructured.

        affinity : string or callable, default: "euclidean"
            Metric used to compute the linkage. Can be "euclidean", "l1", "l2",
            "manhattan", "cosine", or 'precomputed'.
            If linkage is "ward", only "euclidean" is accepted.

        memory : Instance of joblib.Memory or string (optional)
            Used to cache the output of the computation of the tree.
            By default, no caching is done. If a string is given, it is the
            path to the caching directory.

        n_components : int (optional)
            Number of connected components. If None the number of connected
            components is estimated from the connectivity matrix.
            NOTE: This parameter is now directly determined from the
            connectivity matrix and will be removed in 0.18

        compute_full_tree : bool or 'auto' (optional)
            Stop early the construction of the tree at n_clusters. This is
            useful to decrease computation time if the number of clusters is
            not small compared to the number of samples. This option is
            useful only when specifying a connectivity matrix. Note also that
            when varying the number of clusters and using caching, it may
            be advantageous to compute the full tree.

        linkage : {"ward", "complete", "average"}, optional, default: "ward"
            Which linkage criterion to use. The linkage criterion determines
            which distance to use between sets of observation. The algorithm
            will merge the pairs of cluster that minimize this criterion.

            - ward minimizes the variance of the clusters being merged.
            - average uses the average of the distances of each observation of
              the two sets.
            - complete or maximum linkage uses the maximum distances between
              all observations of the two sets.

        pooling_func : callable, default=np.mean
            This combines the values of agglomerated features into a single
            value, and should accept an array of shape [M, N] and the keyword
            argument ``axis=1``, and reduce it to an array of size [M].

        Attributes
        ----------
        labels_ : array [n_samples]
            cluster labels for each point

        n_leaves_ : int
            Number of leaves in the hierarchical tree.

        n_components_ : int
            The estimated number of connected components in the graph.

        children_ : array-like, shape (n_nodes-1, 2)
            The children of each non-leaf node. Values less than `n_samples`
            correspond to leaves of the tree which are the original samples.
            A node `i` greater than or equal to `n_samples` is a non-leaf
            node and has children `children_[i - n_samples]`. Alternatively
            at the i-th iteration, children[i][0] and children[i][1]
            are merged to form node `n_samples + i`

        """
        super(AgglomerativeClustering, self). __init__(
            n_clusters, affinity,
            memory, connectivity, n_components,
            compute_full_tree, linkage,
            pooling_func, return_distance)

    def fit(self, X, **kwargs):
        """Fit the hierarchical clustering on the data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The samples a.k.a. observations.

        Returns
        -------
        self
        """
        if self.n_clusters == 'auto':
            # assign an arbitrary high number for the max number of clusters
            self.n_clusters = int(.75 * X.shape[0])
        super(AgglomerativeClustering, self).fit(X, **kwargs)
        try:
            # use self.distances
            # TODO
            raise NotImplementedError()
        except AttributeError:
            logging.error("Automatic discovery of the number of clusters "
                          "cannot be performed. AgglomerativeClustering from "
                          "adenine.external does not contain a "
                          "`self.distances` attribute. Try to update adenine.")
        # hence, when optimal_clusters is defined, use it
        optimal_clusters = -1  # TODO
        self.n_clusters = optimal_clusters
        # perform the standard fit
        super(AgglomerativeClustering, self).fit(X, **kwargs)
