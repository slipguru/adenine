#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
import warnings

import numpy as np

from sklearn.preprocessing import Imputer
from sklearn.neighbors import NearestNeighbors

class DummyNone:
    """Dummy class that does nothing.

    It is a sklearn 'transforms', it implements both a fit and a transform method and it just returns the data in input. It has been created only for consistency with sklearn.
    """
    def __init__(self):
        pass

    def fit(self,X):
        return self

    def transform(self,X):
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
            if y:
                super(Imputer, self).fit(X,y)
            else:
                super(Imputer, self).fit(X)
        return self


    def transform(self,X):
        if self.strategy.lower() in ['nearest_neighbors', 'nn']:
            _X = X[self._mask,:].copy()
            _X[self.missing] = self.statistics_[self.missing]
            return _X
        else:
            return super(Imputer, self).transform(X)


    def _get_mask(self, X, value_to_mask):
        """Compute the boolean mask X == missing_values. [copy/pasted from sklearn.preprocessing]"""
        if value_to_mask == "NaN" or np.isnan(value_to_mask):
            return np.isnan(X)
        else:
            return X == value_to_mask

    def _get_row_indexes(self, i, c_idx):
        """
        Get which samples do not have missing values or have the same missing value as the i-th sample.
        """
        # Drop the column with missing values in the i-th sample
        _missing = self.missing.copy()
        _missing = _missing[:,c_idx]

        # Get the filled columns
        r_idx = []
        for k, r in enumerate(_missing):
            _not_row = [not j for j in r]
            if np.prod(_not_row, dtype=np.bool): # it's like False is not in _not_row
                r_idx.append(k)

        return np.array(r_idx)

    def _filling_worker(self, X, row, i):
        """
        Worker for parallel execution of self._nn_fit()
        """

        # the list of non-missing values for the i-th row
        c_idx = np.where([not j for j in row])[0]

        # Generate the training matrix (only the non-empty columns)
        r_idx = self._get_row_indexes(i, c_idx)
        Xtr = X[r_idx[:,np.newaxis],c_idx] # get the full matrix of possible neighbors

        neigh = NearestNeighbors(n_neighbors=min(6,Xtr.shape[0]), n_jobs=1)

        # Get the nearest Neighbors
        neigh.fit(Xtr)

        with warnings.catch_warnings(): # shut-up deprecation warnings
            warnings.simplefilter("ignore")
            _nn_idx = neigh.kneighbors(X[i,c_idx], return_distance=False)

        _nn_idx = _nn_idx[0]

        # Evaluate the average of the nearest Neighbors
        neighbors = X[r_idx[_nn_idx[1:]],:] # matrix of nearest neighbors, skip the first one which is the same

        with warnings.catch_warnings(): # shut-up deprecation warnings
            warnings.simplefilter("ignore")
            _nanmean = np.nanmean(neighbors[:,np.where(row)[0]], axis=0)

        self.statistics_[i,row] = _nanmean

    def _nn_fit(self, X):
        """Impute the input data matrix using the Nearest Neighbors approach.

        This implementation follows, approximately, the strategy proposed in: [Hastie, Trevor, et al. "Imputing missing data for gene expression arrays." (1999): 1-7.]
        """
        # 1. Find missing values

        self.missing = self._get_mask(X, self.missing_values)
        # 2. Drop empty rows (I cannot deal with that)
        _mask = np.prod(self.missing, axis=1, dtype=np.bool)
        self._mask = np.array([not j for j in _mask])
        _X = X[self._mask,:].copy()
        self.missing = self.missing[self._mask,:]

        # 3. Statistics init
        self.statistics_ = np.empty_like(_X)

        # 4. For each row that presents a True value in missing:
        #       drop the True column and get the first K Nearest Neighbors
        _cond = True
        count = 0
        while _cond and count < 100:
            for i, row in enumerate(self.missing):
                if row.any(): # i.e. if True in row:
                    self._filling_worker(_X, row, i)
            _cond = np.isnan(self.statistics_).any()
            _X[self.missing] = self.statistics_[self.missing]
            count += 1

        # Log the failure
        if _cond:
            logging.info("Data imputing partially failed.")

        return self
