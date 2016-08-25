#!/usr/bin/env python
# coding=utf-8

# select the models

import os
import sys
import numpy as np
from math import sqrt

from collections import Counter

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_random_state
from sklearn.metrics import f1_score, r2_score, roc_auc_score
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.cross_validation import StratifiedKFold


def _rmse(y, y_pred):
    """
    return the rmse
    :param y:
    :param y_pred:
    :return:
    """

    return np.sqrt(mean_squared_error(y, y_pred))


def _bootstraps(n, rs):
    """
    return bootstrap smaple indices for given n
    :param n:
    :param rs:
    :return:
    """

    bs_inds = rs.randint(n, size=n)
    return bs_inds, np.setdiff1d(range(n), bs_inds)


class EnsembleSelectRegressor(BaseEstimator, RegressorMixin):
    """
    Parameters:
    -----------
    db_file : string

    model: list or None

    n_best: int(default : 5)
        number of top models in initial ensemble

    n_folds : int(default : 3)

    bag_fraction : float(default : 0.25)
        Fraction of (post-pruning) models to randomly select for each bag.

    prune_fraction : float(default:0.8)
        Fraction of worst models to prune before ensemble selection.

    score_metrics : rmse

    epsilon : float(default: 0.0001)
        minimum score improvement to add model to ensemble.
        ignored if user_epsilon is False.

    max_models : int(default:50)
        maximum number of models to include in an ensemble

    verbose : boolean(default:False)
        turn on verbose messgaes

    use_bootstrap : boolean (default:False)

    use_epsilon : boolean (default : False)

    random_state : int, randomstate instance or None (default : None)


    """

    _metrics = {
        'rmse': _rmse,
    }

    def __init__(self, db_file=None,
                 models=None, n_best=5, n_folds=3,
                 n_bags=20, bag_fraction=0.25,
                 prune_fraction=0.8, score_metric='rmse',
                 epsilon=0.0001, max_models=50,
                 use_epsilon=False, use_bootstrap=False,
                 verbose=False, random_state=None):

        self.db_file = db_file
        self.models = models
        self.n_best = n_best
        self.n_folds = n_folds
        self.n_bags = n_bags
        self.bag_fraction = bag_fraction
        self.prune_fraction = prune_fraction
        self.score_metric = score_metric
        self.epsilon = epsilon
        self.max_models = max_models
        self.use_epsilon = use_epsilon
        self.use_bootstrap = use_bootstrap
        self.verbose = verbose
        self.random_state = random_state

        self._check_parms()

        self._folds = None
        self._n_models = 0
        self._metric = None
        self._ensemble = Counter()
        self._model_scores = []
        self._scored_models = []
        self._fitted_models = []

    def _check_parms(self):
        """
        parameter check
        :return:
        """

        # if (not self.db_file):

        if self.epsilon < 0.0:
            msg = "epsilon must be >= 0.0"
            raise ValueError(msg)

        metric_names = self._metrics.keys()
        if self.score_metric not in metric_names:
            msg = "score_metric not in %s " % metric_names
            return ValueError(msg)

        if self.n_best < 1:
            msg = "n_best must be >= 1."
            return ValueError(msg)

        if self.max_models < self.n_folds:
            msg = "max_models must be >= n_best"
            raise ValueError(msg)

        if not self.use_bootstrap:
            if self.n_folds < 2:
                msg = "n_folds must be >= 2 for StratifiedKFolds."
                raise ValueError(msg)

        else:
            if self.n_folds < 1:
                msg = "n_folds must be >= 1 with bootstrap"
                raise ValueError(msg)

    def fit(self, X, y):
        """
        perform model fitting and ensemble building.
        :param X:
        :param y:
        :return:
        """

        self.fit_models(X, y)
        self.build_ensemble(X, y)

        return self

    def fit_models(self, X, y):
        """
        perform internal cross-validation fit
        :param X:
        :param y:
        :return:
        """

        if self.verbose:
            sys.stderr.write("\nfitting models:  \n")

        if self.use_bootstrap:
            n = X.shape[0]
            rs = check_random_state(self.random_state)

            self._folds = [_bootstraps(n, rs) for _ in range(self.n_folds)]
        else:
            self._folds = list(StratifiedKFold(y, n_folds=self.n_folds))


        for model_idx in range(self._n_models):


            model_folds = []

































































