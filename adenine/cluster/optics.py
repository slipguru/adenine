"""Clustering OPTICS (Ordering Points To Identify Clustering Structure).

Based on article description:
     - M.Ankerst, M.Breunig, H.Kriegel, J.Sander. OPTICS: Ordering Points To
     Identify the Clustering Structure. 1999.

Original:
@authors Andrei Novikov (pyclustering@yandex.ru)
@date 2014-2017
@copyright GNU Public License

@cond GNU_PUBLIC_LICENSE
    PyClustering is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PyClustering is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
@endcond

Modification by:
Authors: Federico Tomasi, federico.tomasi@dibris.unige.it
"""
import matplotlib.pyplot as plt
import numpy as np
import itertools

from scipy.spatial import distance
from scipy.sparse import issparse

from sklearn.utils.fixes import partial
from sklearn.metrics.pairwise import check_pairwise_arrays
from sklearn.metrics.pairwise import _parallel_pairwise
from sklearn.metrics.pairwise import PAIRWISE_BOOLEAN_FUNCTIONS
from sklearn.metrics.pairwise import PAIRWISE_DISTANCE_FUNCTIONS

from sklearn.metrics import pairwise_distances
from sklearn.base import BaseEstimator, ClusterMixin

# TODO remove dependencies
import pyclustering.core.optics_wrapper as wrapper


def _pairwise_callable(X, Y, metric, **kwds):
    """Handle the callable case for pairwise_{distances,kernels}
    """
    try:
        X, Y = check_pairwise_arrays(X, Y)
    except TypeError:
        X, Y = check_pairwise_arrays(X, Y, dtype=object)  # try not to convert

    if X is Y:
        # Only calculate metric for upper triangle
        out = np.zeros((X.shape[0], Y.shape[0]), dtype='float')
        iterator = itertools.combinations(range(X.shape[0]), 2)
        for i, j in iterator:
            out[i, j] = metric(X[i], Y[j], **kwds)

        # Make symmetric
        # NB: out += out.T will produce incorrect results
        out = out + out.T

        # Calculate diagonal
        # NB: nonzero diagonals are allowed for both metrics and kernels
        for i in range(X.shape[0]):
            x = X[i]
            out[i, i] = metric(x, x, **kwds)

    else:
        # Calculate all cells
        out = np.empty((X.shape[0], Y.shape[0]), dtype='float')
        iterator = itertools.product(range(X.shape[0]), range(Y.shape[0]))
        for i, j in iterator:
            out[i, j] = metric(X[i], Y[j], **kwds)

    return out


_VALID_METRICS = ['euclidean', 'l2', 'l1', 'manhattan', 'cityblock',
                  'braycurtis', 'canberra', 'chebyshev', 'correlation',
                  'cosine', 'dice', 'hamming', 'jaccard', 'kulsinski',
                  'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
                  'russellrao', 'seuclidean', 'sokalmichener',
                  'sokalsneath', 'sqeuclidean', 'yule', "wminkowski"]


def pairwise_distances(X, Y=None, metric="euclidean", n_jobs=1, **kwds):
    """ Compute the distance matrix from a vector array X and optional Y.

    This method takes either a vector array or a distance matrix, and returns
    a distance matrix. If the input is a vector array, the distances are
    computed. If the input is a distances matrix, it is returned instead.

    This method provides a safe way to take a distance matrix as input, while
    preserving compatibility with many other algorithms that take a vector
    array.

    If Y is given (default is None), then the returned matrix is the pairwise
    distance between the arrays from both X and Y.

    Valid values for metric are:

    - From scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
      'manhattan']. These metrics support sparse matrix inputs.

    - From scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
      'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
      'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
      'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
      See the documentation for scipy.spatial.distance for details on these
      metrics. These metrics do not support sparse matrix inputs.

    Note that in the case of 'cityblock', 'cosine' and 'euclidean' (which are
    valid scipy.spatial.distance metrics), the scikit-learn implementation
    will be used, which is faster and has support for sparse matrices (except
    for 'cityblock'). For a verbose description of the metrics from
    scikit-learn, see the __doc__ of the sklearn.pairwise.distance_metrics
    function.

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : array [n_samples_a, n_samples_a] if metric == "precomputed", or, \
             [n_samples_a, n_features] otherwise
        Array of pairwise distances between samples, or a feature array.

    Y : array [n_samples_b, n_features], optional
        An optional second feature array. Only allowed if metric != "precomputed".

    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by scipy.spatial.distance.pdist for its metric parameter, or
        a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
        If metric is "precomputed", X is assumed to be a distance matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them.

    n_jobs : int
        The number of jobs to use for the computation. This works by breaking
        down the pairwise matrix into n_jobs even slices and computing them in
        parallel.

        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    `**kwds` : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a scipy.spatial.distance metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.

    Returns
    -------
    D : array [n_samples_a, n_samples_a] or [n_samples_a, n_samples_b]
        A distance matrix D such that D_{i, j} is the distance between the
        ith and jth vectors of the given matrix X, if Y is None.
        If Y is not None, then D_{i, j} is the distance between the ith array
        from X and the jth array from Y.

    """
    if (metric not in _VALID_METRICS and
            not callable(metric) and metric != "precomputed"):
        raise ValueError("Unknown metric %s. "
                         "Valid metrics are %s, or 'precomputed', or a "
                         "callable" % (metric, _VALID_METRICS))

    if metric == "precomputed":
        X, _ = check_pairwise_arrays(X, Y, precomputed=True)
        return X
    elif metric in PAIRWISE_DISTANCE_FUNCTIONS:
        func = PAIRWISE_DISTANCE_FUNCTIONS[metric]
    elif callable(metric):
        func = partial(_pairwise_callable, metric=metric, **kwds)
    else:
        if issparse(X) or issparse(Y):
            raise TypeError("scipy distance metrics do not"
                            " support sparse matrices.")

        dtype = bool if metric in PAIRWISE_BOOLEAN_FUNCTIONS else None

        X, Y = check_pairwise_arrays(X, Y, dtype=dtype)

        if n_jobs == 1 and X is Y:
            return distance.squareform(distance.pdist(X, metric=metric,
                                                      **kwds))
        func = partial(distance.cdist, metric=metric, **kwds)

    return _parallel_pairwise(X, Y, func, n_jobs, **kwds)


class ordering_visualizer:
    """!
    @brief Cluster ordering diagram visualizer that represents dataset
    graphically as density-based clustering structure.

    @see ordering_analyser

    """

    @staticmethod
    def show_ordering_diagram(analyser, amount_clusters=None):
        """Display cluster-ordering diagram.

        @param[in] analyser (ordering_analyser): cluster-ordering analyser
        whose ordering diagram should be displayed.
        @param[in] amount_clusters (uint): if not 'None' it displays
        connectivity radius line that can used for allocation of specified
        amount of clusters.

        Example demonstrates general abilities of 'ordering_visualizer' class:
        @code
        # Display cluster-ordering diagram with connectivity radius is used
        for allocation of three clusters.
        ordering_visualizer.show_ordering_diagram(analyser, 3);

        # Display cluster-ordering diagram without radius.
        ordering_visualizer.show_ordering_diagram(analyser);
        @endcode

        """
        ordering = analyser.cluster_ordering
        indexes = [i for i in range(0, len(ordering))]

        axis = plt.subplot(111)
        axis.bar(indexes, ordering, color = 'black')
        plt.xlim([0, len(ordering)])

        if (amount_clusters is not None):
            radius = analyser.calculate_connvectivity_radius(amount_clusters);
            plt.axhline(y = analyser.calculate_connvectivity_radius(amount_clusters), linewidth = 2, color = 'b');
            plt.text(0, radius + radius * 0.03, " Radius:   " + str(round(radius, 4)) + ";\n Clusters: " + str(amount_clusters), color = 'b', fontsize = 10);

        plt.show();


class ordering_analyser:
    """Analyser of cluster ordering diagram.

    Using cluster-ordering it is able to connectivity radius for allocation
    of specified amount of clusters and calculate amount of clusters using
    specified connectivity radius. Cluster-ordering is formed by OPTICS
    during cluster analysis.
    """

    @property
    def cluster_ordering(self):
        """!
        @brief (list) Returns values of dataset cluster ordering.

        """
        return self._ordering;


    def __init__(self, ordering_diagram):
        """!
        @brief Analyser of ordering diagram that is based on reachability-distances.

        @see calculate_connvectivity_radius

        """
        self._ordering = ordering_diagram;


    def __len__(self):
        """!
        @brief Returns length of clustering-ordering diagram.

        """
        return len(self._ordering);


    def calculate_connvectivity_radius(self, amount_clusters, maximum_iterations = 100):
        """!
        @brief Calculates connectivity radius of allocation specified amount of clusters using ordering diagram.

        @param[in] amount_clusters (uint): amount of clusters that should be allocated by calculated connectivity radius.
        @param[in] maximum_iterations (uint): maximum number of iteration for searching connectivity radius to allocated specified amount of clusters (by default it is restricted by 100 iterations).

        @return (double) Value of connectivity radius, it may be 'None' if connectivity radius hasn't been found for the specified amount of iterations.

        """

        maximum_distance = max(self._ordering);

        upper_distance = maximum_distance;
        lower_distance = 0.0;

        radius = None;
        result = None;

        if (self.extract_cluster_amount(maximum_distance) <= amount_clusters):
            for _ in range(maximum_iterations):
                radius = (lower_distance + upper_distance) / 2.0;

                amount = self.extract_cluster_amount(radius);
                if (amount == amount_clusters):
                    result = radius;
                    break;

                elif (amount == 0):
                    break;

                elif (amount > amount_clusters):
                    lower_distance = radius;

                elif (amount < amount_clusters):
                    upper_distance = radius;

        return result;


    def extract_cluster_amount(self, radius):
        """!
        @brief Obtains amount of clustering that can be allocated by using specified radius for ordering diagram.
        @details When growth of reachability-distances is detected than it is considered as a start point of cluster,
                 than pick is detected and after that recession is observed until new growth (that means end of the
                 current cluster and start of a new one) or end of diagram.

        @param[in] radius (double): connectivity radius that is used for cluster allocation.


        @return (unit) Amount of clusters that can be allocated by the connectivity radius on ordering diagram.

        """

        amount_clusters = 1;

        cluster_start = False;
        cluster_pick = False;
        total_similarity = True;
        previous_cluster_distance = None;
        previous_distance = None;

        for distance in self._ordering:
            if (distance >= radius):
                if (cluster_start is False):
                    cluster_start = True;
                    amount_clusters += 1;

                else:
                    if ((distance < previous_cluster_distance) and (cluster_pick is False)):
                        cluster_pick = True;

                    elif ((distance > previous_cluster_distance) and (cluster_pick is True)):
                        cluster_pick = False;
                        amount_clusters += 1;

                previous_cluster_distance = distance;

            else:
                cluster_start = False;
                cluster_pick = False;

            if ( (previous_distance is not None) and (distance != previous_distance) ):
                total_similarity = False;

            previous_distance = distance;

        if ( (total_similarity is True) and (previous_distance > radius) ):
            amount_clusters = 0;

        return amount_clusters;


class OpticsDescriptor(object):
    """Object description used by OPTICS algorithm for cluster analysis."""

    def __init__(self, index=0, core_distance=None, reachability_distance=None):
        """Constructor of object description in optics terms.

        @param[in] index (uint): Index of the object in the data set.
        @param[in] core_distance (double): Core distance that is minimum
        distance to specified number of neighbors.
        @param[in] reachability_distance (double): Reachability distance to
        this object.
        """
        # Reachability distance - the smallest distance to be reachable by
        # core object.
        self.index = index

        # Core distance - the smallest distance to reach specified number
        # of neighbors that is not greater then connectivity radius.
        self.core_distance = core_distance

        # Index of object from the input data.
        self.reachability_distance = reachability_distance

        # True is object has been already traversed.
        self.processed = False

    def __repr__(self):
        return '(%s, [c: %s, r: %s])' % (self.index, self.core_distance, self.reachability_distance);


def optics(X, eps=0.5, min_samples=5, metric='minkowski'):
    pass


class Optics(BaseEstimator, ClusterMixin):
    """Perform Optics clustering from vector array or distance matrix
    OPTICS - Ordering Points To Identify Clustering Structure
    OPTICS is a density-based algorithm. Purpose of the algorithm is to
    provide explicit clusters, but create clustering-ordering representation
    of the input data. Clustering-ordering information contains information
    about internal structures of data set in terms of density and proper
    connectivity radius can be obtained for allocation required amount of
    clusters using this diagram. In case of usage additional input parameter
    'amount of clusters' connectivity radius should be bigger than real -
    because it will be calculated by the algorithms if requested amount of
    clusters is not allocated.

    Example
    -------
    # Create OPTICS algorithm for cluster analysis
    optics_instance = optics(sample, 0.5, 6);

    # Run cluster analysis
    optics_instance.process();

    # Obtain results of clustering
    clusters = optics_instance.get_clusters();
    noise = optics_instance.get_noise();

    # Obtain rechability-distances
    ordering = ordering_analyser(optics_instance.get_ordering());

    # Visualization of cluster ordering in line with reachability distance.
    ordering_visualizer.show_ordering_diagram(ordering);
    """

    def __init__(self, eps=0.5, min_samples=5, n_clusters=None, ccore=False,
                 metric='euclidean'):
        """Constructor of clustering algorithm OPTICS.

        Parameters
        ----------
        eps : float, optional
            The maximum distance between two samples for them to be considered
            as in the same neighborhood.

        min_samples : int, optional
            Minimum number of shared neighbors required for establishing links
            between points.

        n_clusters : int, optional
            Allocate fixed number of clusters.

        ccore : bool, optional
            Use DLL CCORE (C++ implementation).

        metric : string, or callable
            The metric to use when calculating distance between instances in a
            feature array. If metric is a string or callable, it must be one of
            the options allowed by metrics.pairwise.calculate_distance for its
            metric parameter.
            If metric is "precomputed", X is assumed to be a distance matrix and
            must be square. X may be a sparse matrix, in which case only "nonzero"
            elements may be considered neighbors for DBSCAN.

            .. versionadded:: 0.17
               metric *precomputed* to accept precomputed sparse matrix.

        """
        self.eps = eps
        self.minpts = min_samples
        self.n_clusters = n_clusters
        self.ccore = ccore
        self.metric = metric
        self._ordering = None

    def get_ordering(self):
        """Clustering ordering information about the input data set.

        Clustering ordering of data-set contains the information about
        the internal clustering structure in line with connectivity radius.

        @return (ordering_analyser) Analyser of clustering ordering.
        """
        if self._ordering is None:
            self._ordering = []

            for cluster in self.clusters_:
                for index in cluster:
                    optics_obj = self._optics_objects[index]
                    if optics_obj.reachability_distance is not None:
                        self._ordering.append(optics_obj.reachability_distance)

        return self._ordering

    def fit(self, X):
        """Cluster analysis in line with rules of OPTICS algorithm.

        Results of clustering can be obtained using corresponding gets methods.
        """
        if not self.eps > 0.0:
            raise ValueError("eps must be positive.")

        self._X_fit = X

        if self.ccore:
            # TODO
            (self.clusters_, self.noise_, self._ordering, self.eps) = \
                wrapper.optics(self._X_fit, self.eps, self.minpts,
                               self.n_clusters)

        else:
            self._fit(X)

        labels = np.empty(X.shape[0], dtype=int)
        for i, cluster in enumerate(self.clusters_):
            cluster = np.array(cluster, dtype=int)
            labels[cluster] = i

        # noise do not belong to any cluster, so they belong to their own
        len_clusters = len(self.clusters_)
        for i, cluster in enumerate(self.noise_):
            labels[cluster] = i + len_clusters

        self.labels_ = labels
        return self

    def allocate_clusters(self, X):
        self._processed = np.zeros(X.shape[0], dtype=bool)  # none is processed
        # List of OPTICS objects that corresponds to objects from input sample
        self._optics_objects = [OpticsDescriptor(i) for i in range(X.shape[0])]
        self._ordered_database = []  # List of OPTICS objects in traverse order

        for optics_obj in self._optics_objects:
            if not optics_obj.processed:
                _expand_cluster_order(
                    self._ordered_database, optics_obj, self._optics_objects,
                    self.minpts, X, self.metric, self.eps)

        # _clusters : list of clusters where each cluster contains indexes
        # of objects from input data
        # _noise : List of allocated noise objects
        self.clusters_ = []
        self.noise_ = []

        current_cluster = []
        for optics_obj in self._ordered_database:
            if optics_obj.reachability_distance is None or \
                    optics_obj.reachability_distance > self.eps:
                if optics_obj.core_distance is not None and \
                        optics_obj.core_distance <= self.eps:
                    if len(current_cluster) > 0:
                        self.clusters_.append(current_cluster)
                        current_cluster = []

                    current_cluster.append(optics_obj.index)
                else:
                    self.noise_.append(optics_obj.index)
            else:
                current_cluster.append(optics_obj.index)

        if len(current_cluster) > 0:
            self.clusters_.append(current_cluster)

    def _fit(self, X):
        # Performs cluster allocation and builds ordering diagram based on
        # reachability-distances.
        self.allocate_clusters(X)

        if self.n_clusters is not None and self.n_clusters != len(self.clusters_):
            analyser = ordering_analyser(self.get_ordering())
            radius = analyser.calculate_connvectivity_radius(self.n_clusters)
            if radius is not None:
                self.eps = radius
                self.allocate_clusters(X)


def _expand_cluster_order(ordered_db, optics_obj, optics_objs, minpts,
                          X, metric, eps):
    """Expand cluster order from not processed optic-object.

    Traverse procedure is performed until objects are reachable from
    core-objects in line with connectivity radius.
    Order database is updated during expanding.

    Parameters
    ----------
    optics_obj : object
        Object that hasn't been processed.
    """
    optics_obj.processed = True

    neighbors = _neighbor_indexes(
        optics_obj.index, X, metric, eps)
    optics_obj.reachability_distance = None

    ordered_db.append(optics_obj)

    # Check core distance
    if len(neighbors) >= minpts:
        # neighbors.sort(key=lambda obj: obj[1])
        neighbors = neighbors[neighbors[:, 1].argsort()]
        optics_obj.core_distance = neighbors[minpts - 1][1]

        # Continue processing
        order_seed = list()
        _update_order_seed(
            optics_objs, optics_obj, neighbors, order_seed)

        while len(order_seed) > 0:
            optic_descriptor = order_seed[0]
            order_seed.remove(optic_descriptor)

            neighbors = _neighbor_indexes(
                optic_descriptor.index, X, metric, eps)
            optic_descriptor.processed = True

            ordered_db.append(optic_descriptor)

            if len(neighbors) >= minpts:
                # neighbors.sort(key=lambda obj: obj[1])
                neighbors = neighbors[neighbors[:, 1].argsort()]
                optic_descriptor.core_distance = neighbors[minpts - 1][1]

                _update_order_seed(
                    optics_objs,
                    optic_descriptor, neighbors, order_seed)
            else:
                optic_descriptor.core_distance = None

    else:
        optics_obj.core_distance = None

def _update_order_seed(optics_objs, optic_descriptor, neighbors,
                       order_seed):
    """Update sorted list of reachable objects (from core-object) that
    should be processed using neighbors of core-object.

    @param[in] optic_descriptor (OpticsDescriptor): Core-object whose neighbors should be analysed.
    @param[in] neighbors (list): List of neighbors of core-object.
    @param[in|out] order_seed (list): List of sorted object in line with reachable distance.

    """
    for idx, current_reachable_distance in neighbors:
        idx = int(idx)
        if not optics_objs[idx].processed:
            reachable_distance = max(
                current_reachable_distance, optic_descriptor.core_distance)
            if optics_objs[idx].reachability_distance is None:
                optics_objs[idx].reachability_distance = reachable_distance

                # insert element in queue O(n) - worst case.
                index_insertion = len(order_seed)
                for index_seed in range(0, len(order_seed)):
                    if (reachable_distance < order_seed[index_seed].reachability_distance):
                        index_insertion = index_seed
                        break

                order_seed.insert(index_insertion, optics_objs[idx])

            else:
                if (reachable_distance < optics_objs[idx].reachability_distance):
                    optics_objs[idx].reachability_distance = reachable_distance
                    order_seed.sort(key=lambda obj: obj.reachability_distance)


def _neighbor_indexes(index, X, metric, eps):
    """List of indices and distance of neighbors of a point."""
    # get neighbors of index
    sample = X[index]
    if isinstance(sample, np.ndarray):
        # avoid deprecation warning
        sample = sample.reshape(1, -1)

    neigh = pairwise_distances(sample, X, metric)
    idx = np.where(neigh <= eps)[1]

    # discard point itself
    idx = idx[idx != index]
    return np.vstack((idx, neigh.flat[idx])).T
