"""Agglomerative clustering class extension."""
import logging
import numpy as np
from sklearn.externals.joblib import Memory
from adenine.externals import AgglomerativeClustering


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
