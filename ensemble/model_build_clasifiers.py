#!/usr/bin/env python
# coding=utf-8

# model library


from __future__ import print_function

import numpy as np

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.pipeline import Pipeline
from sklearn.grid_search import ParameterGrid
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
# 这个是干什么的
from sklearn.kernel_approximation import Nystroem


# generic model bulider
def build_model(model_class, param_grid):
    print("Building %s model" %(str(model_class).split(".")[-1][:-2]))
    return [model_class(**p) for p in ParameterGrid(param_grid)]


def build_randomForestClassifiers(random_state=None):

    param_grid = {
        "n_estimators": [20, 50, 100],
        "criterion": ['gini', 'entropy'],
        "max_features": [None, 'auto', 'sqrt', 'log2'],
        "max_depth": [1, 2, 5, 10],
        "min_density": [0.25, 0.5, 0.75, 1.0],
        "random_state": [random_state],
    }

    return build_model(RandomForestClassifier(), param_grid)


def build_gradientBoostingClassifiers(random_state=None):

    param_grid = {
        "n_estimators": [10, 20, 50, 100],
        "max_features": np.linspace(0.2, 1.0, 5),
        "max_depth": [1, 2, 5, 10],
        "subsmaple": np.linspace(0.2, 1.0, 5),
        "random_state": [random_state],
    }

    return build_model(GradientBoostingClassifier(), param_grid)



























