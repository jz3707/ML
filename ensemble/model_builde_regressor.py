#!/usr/bin/env python
# coding=utf-8

# model library


from __future__ import print_function

import numpy as np

from sklearn.utils import check_random_state

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.pipeline import Pipeline
from sklearn.grid_search import ParameterGrid
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LogisticRegression

from sklearn.cluster import KMeans
# 这个是干什么的
# from sklearn.kernel_approximation import Nystroem


# generic model bulider
def build_model(model_class, param_grid):
    print("Building %s model" %(str(model_class).split(".")[-1][:-2]))
    return [model_class(**p) for p in ParameterGrid(param_grid)]


def build_randomForestRegressor(random_state=None):

    param_grid = {
        "n_estimators": [20, 50, 100],
        "criterion": ['gini', 'entropy'],
        "max_features": [None, 'auto', 'sqrt', 'log2'],
        "max_depth": [1, 2, 5, 10],
        "min_density": [0.25, 0.5, 0.75, 1.0],
        "random_state": [random_state],
    }

    return build_model(RandomForestRegressor, param_grid)


def build_gradientBoostingRegressor(random_state=None):

    param_grid = {
        "n_estimators": [10, 20, 50, 100],
        "max_features": np.linspace(0.2, 1.0, 5),
        "max_depth": [1, 2, 5, 10],
        "subsmaple": np.linspace(0.2, 1.0, 5),
        "random_state": [random_state],
    }

    return build_model(GradientBoostingRegressor, param_grid)


def build_sgdRegressor(random_state=None):

    param_grid = {
        "loss": ["log", "modified_huber"],
        "penalty": ["elasticnet"],
        "alpha": [0.0001, 0.001, 0.01, 0.1],
        "learning_rate": ['constant', "optimal"],
        "n_iter": [2, 5, 10],
        "eta0": [0.001, 0.01, 0.1],
        "l1_ratio": np.linspace(0.0, 1.0, 3)

    }

    return build_model(SGDRegressor, param_grid)


def build_decisionTreeRegressor(random_state=None):

    rs = check_random_state(random_state)

    param_grid = {
        "criterion": ['gini', 'entropy'],
        "max_features": [None, 'auto', 'sqrt', 'log2'],
        "max_depth": [None, 1, 2, 5, 10],
        "min_samples_split": [1, 2, 5, 10],
        "random_state": [rs.random_integers(100000) for i in range(3)],
    }

    return build_model(DecisionTreeRegressor, param_grid)


def build_extraTreesRegressor(random_state=None):

    param_grid = {
        "criterion": ['gini', 'entropy'],
        "n_estimators": [5, 10, 20],
        "max_features": [None, 'auto', 'sqrt', 'log2'],
        "max_depth": [None, 1, 2, 5, 10],
        "min_samples_split": [2, 5, 10],
        "random_state": [random_state],
    }

    return build_model(ExtraTreesRegressor, param_grid)


def build_logisticRegression(random_state=None):

    param_grid = {
        "penalty": ['l1', 'l2'],
        "max_iter": [5, 10, 20],
        "solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
        "C": [0.001, 0.01, 0.1],
    }

    return build_model(LogisticRegression, param_grid)


def build_kmeansPipelines(random_state=None):

    print("Building KMeans-Logistic Regression Pipelines.")

    param_grid = {
        "n_clusters": range(5, 205, 5),
        "init": ["k-means++", 'random'],
        "n_int": [1, 2, 5, 10],
        "random_state": [random_state],

    }

    models = []

    for param in ParameterGrid(param_grid):
        km = KMeans(**param)
        lr = LogisticRegression()
        models.append(Pipeline([('km', km), ('lr', lr)]))

    return models


models_dict = {
    "forest": build_randomForestRegressor,
    "gbr": build_gradientBoostingRegressor,
    "sgd": build_sgdRegressor,
    "dtree": build_decisionTreeRegressor,
    "extra": build_extraTreesRegressor,
    "log": build_logisticRegression,
    "kmp": build_kmeansPipelines
}

def build_model_library(model_types=['dtree'], random_seed=None):
    models = []
    for m in model_types:
        models.extend(models_dict[m](random_state=random_seed))
    return models
















