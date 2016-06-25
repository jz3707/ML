#!/usr/bin/env python
# coding=utf-8

import math
import copy
import numpy as np
import matplotlib.pyplot as plt

isDebug = False


# 指定k个Gaussian分布，这里k=2，注意2个Gaussian具有相同的均方差Sigma，均值分别为Mu1，Mu2
def init_data(sigma, mu1, mu2, k, n):
    global X
    global Mu
    global Expec

    # 初始化X，为
    X = np.zeros((1, n))
    Mu = np.random.random(2)
    Expec = np.zeros((n, k))

    for i in xrange(0, n):
        if np.random.random(1) > 0.5:
            X[0, i] = np.random.normal() * sigma + mu1
        else:
            X[0, i] = np.random.normal() * sigma + mu2
    if isDebug:
        print("***************")
        print(u"初始化观测数据X：")
        print(X)


# E-step 计算E[zij]
def e_step(sigma, k, n):
    global Expec
    global Mu
    global X

    for i in xrange(0, n):
        denom = 0
        for j in xrange(0, k):
            denom += math.exp((-1 / (2 * (float(sigma ** 2)))) * (float(X[0, i] - Mu[j])) ** 2)
        for j in xrange(0, k):
            numer = math.exp((-1 / (2 * (float(sigma ** 2)))) * (float(X[0, i] - Mu[j])) ** 2)
            Expec[i, j] = numer / denom

    if isDebug:
        print("***************")
        print(u"隐变量E(Z)：")
        print(Expec)


# M-step 最大化E[zij]的参数Mu
def m_step(k, n):
    global Expec
    global X

    for j in xrange(0, k):
        numer = 0
        denom = 0
        for i in xrange(0, n):
            numer += Expec[i, j] * X[0, j]
            denom += Expec[i, j]

        Mu[j] = numer / denom


def run(sigma, mu1, mu2, k, n, iter_num, epsilon):
    init_data(sigma, mu1, mu2, k, n)
    print("初始化<u1, u2> : ", Mu)
    for i in range(iter_num):
        oldmu = copy.deepcopy(Mu)
        e_step(sigma, k, n)
        m_step(k, n)
        print(i, Mu)
        if sum(abs(Mu - oldmu) < epsilon):
            break


if __name__ == '__main__':
    run(6, 40, 20, 2, 1000, 1000, 0.0001)
    plt.hist(X[0, :], 40)
    plt.show()
