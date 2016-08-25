#!/usr/bin/env python
# coding=utf-8

# function of loading data


import numpy as np


def read_data(file_name):
    """

    :param file_name:
    :return:
    """
    f = open(file_name)
    f.readline()
    samples = []
    for line in f:
        line = line.strip().split(",")
        sample = [float(x) for x in line]
        samples.append(sample)

    return samples


def load(train_path, test_path):
    print("Loading data..")
    train = read_data(train_path)
    y_train = np.array([x[0] for x in train])
    X_train = np.array([x[1:] for x in train])
    X_test = np.array(read_data(test_path))
    return X_train, y_train, X_test





