#!/usr/bin/env python
# coding=utf-8

# ensemble stacked
# blending RF, ExtraTrees, GradientBoosting
# 5-folds, train the models, then use the results of the models as variables
# in logistic regression over the validation data of that fold.


from __future__ import division

import numpy as np
np.random.seed(1337)

from . import load_data

from sklearn.metrics import mean_squared_error

from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression


def mseloss(attempt, actual, epsilon=1.0e-15):
    """
    mseloss, score of regressor
    :param attempt:
    :param actual:
    :param epsilon:
    :return:
    """

    attempt = np.clip(attempt, epsilon, 10 - epsilon)
    return mean_squared_error(actual, attempt)


if __name__ == '__main__':

    n_folds = 10
    verbose = True
    shuffle = False

    X, y, X_submission = load_data.load("", "")

    if shuffle:
        # 对y.size进行随机排序
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]

    skf = list(StratifiedKFold(y, n_folds))

    clfs = [RandomForestRegressor(n_estimators=100, n_jobs=-1, criterion='gini'),
            RandomForestRegressor(n_estimators=100, n_jobs=-1, criterion='entropy'),
            ExtraTreesRegressor(n_estimators=100, n_jobs=-1, criterion='gini'),
            ExtraTreesRegressor(n_estimators=100, n_jobs=-1, criterion='entropy'),
            GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, subsample=0.5, max_depth=6)]

    print("Creating train and test sets of blending")

    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):
        print("j : ", j, "clf : ", clf)

        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            print("Fold : ", i)
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]

            clf.fit(X_train, y_train)
            y_submisson = clf.predict_proba(X_test)[:, 1]
            dataset_blend_train[test, j] = y_submisson
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:, 1]

        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)

    print()
    print("Blending")

    clf = LogisticRegression()
    clf.fit(dataset_blend_train, y)
    y_submisson = clf.predict_proba(dataset_blend_test)[:, j]

    print("Linear stretch of predictions to [0, 1]")
    y_submisson = (y_submisson - y_submisson.min()) / (y_submisson.max() - y_submisson.min())

    print("Saving Results.")
    tmp = np.vstack([range(1, len(y_submisson) + 1), y_submisson]).T













































