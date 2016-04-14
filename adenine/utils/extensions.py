#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
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
    def fit(self,X,y=None):
        if self.strategy.lower() in ['nearest_neighbors', 'nn']:
            self.statistics_ = self._nn_fit(X)
        else:
            if y:
                super(Imputer, self).fit(X,y)
            else:
                super(Imputer, self).fit(X)
        return self

    def transform(self, X):
        if self.strategy.lower() in ['nearest_neighbors', 'nn']:
            pass # transform goes here
        else:
            return super(Imputer, self).transform(X)

    def _nn_fit(self,X):
        """Impute the input data matrix using the Nearest Neighbors approach.

        This implementation follows, approximately, the strategy proposed in: [Hastie, Trevor, et al. "Imputing missing data for gene expression arrays." (1999): 1-7.]
        """

        # 1. Find missing values
        # missing = np.isnan(X)
        missing = super(Imputer, self)._get_mask(X)
        print missing










        # asddsa
