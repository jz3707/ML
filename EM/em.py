#!/usr/bin/env python
# coding=utf-8

import math
import numpy as np
import matplotlib.pyplot as plt
import random


# Gaussian function
def gauss(xi, miu, sigma):

    # 注意这里精度不够，特么纠结这么长时间, 原因是因为sigma初始给1.0比较小，所以要给大一点。
    # In[6]: math.exp(-((float(173)-float(120)) / float(1.0)) **2)
    # Out[6]: 0.0
    return math.exp(-((float(xi)-float(miu)) ** 2 / (float(sigma) * 2))) / (
       2 * math.sqrt(math.pi) * float(sigma))


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def gauss_function(xi, a, mu, sigma):
    return a * np.exp(-(xi - mu) ** 2 / (2 * sigma ** 2))


# 更新μk, 第k类均值
def update_muk(x, gamma, nk):
    assert (len(x) == len(gamma))
    return sum([gamma[i] * x[i] for i in xrange(len(x))]) / float(nk)


# 更新σk，第k类的方差，注意这里是平方！！！！
def update_sigma_squarek(x, gamma, muk, nk):
    assert (len(x) == len(gamma))
    return sum([gamma[i] * ((x[i] - muk) ** 2) for i in xrange(len(x))]) / float(nk)


# 比较上次迭代和当前迭代的参数损失
def is_break(cur, now):
    assert (len(cur) == len(now))
    for i in xrange(len(cur)):
        if cur[i] - now[i] > 0.1:
            return False
    return True


def calc_em(samples):
    """

    :type samples: list of samples
    """
    n = len(samples)
    # girl probability
    gp = 0.5
    # boy probability
    bp = 0.5
    # 先验，直接给最大值和最小值
    gmu, gsigma = min(samples), 2
    bmu, bsigma = max(samples), 2

    # parameter set
    cur = [gp, bp, gmu, gsigma, bmu, bsigma]
    now = []

    times = 0
    while times < 100:
        ggamma = range(n)
        bgamma = range(n)
        # i为样本
        i = 0
        # random.shuffle(samples)
        for x in samples:
            ggamma[i] = gp * gauss(x, gmu, gsigma)
            bgamma[i] = bp * gauss(x, bmu, bsigma)
            s = ggamma[i] + bgamma[i]
            # print('x : %f, gguass : %.10f, bgauss : %.10f, s : %.10f, gp : %f, bp : %f, gmu : %f, bmu : %f' % (
            #     x, gauss(x, gmu, gsigma), gauss(x, bmu, bsigma), s, gp, bp, gmu, bmu))
            # print("ggamma[i] : %f\tbgamma[i] : %f\ts : \t %f" % (ggamma[i], bgamma[i], s))

            # 注意公式 γ(i, k) = πkN(xi|μk, σk) / (Σj=1,k πj * n(xi|μj, σj))
            # gp = πk， 选择第k类Gaussian概率, 因为这里只有girl和boy两类，所以s只相加了两个
            ggamma[i] /= s
            bgamma[i] /= s
            i += 1

        # for g in ggamma:
        #     print("g : %f" % g)
        #
        # for b in bgamma:
        #     print("b : %f" % b)
        # gn表示Nk = Σi=1,n γ(i, k)
        gn = sum(ggamma)
        # 更新gp，也就是πk，
        gp = float(gn) / float(n)
        # 同理更新bn，bp
        bn = sum(bgamma)
        bp = float(bn) / float(n)
        # 更新gmu
        gmu = update_muk(samples, ggamma, gn)
        # 更新bmu
        bmu = update_muk(samples, bgamma, bn)
        # 更新gsigma 注意这里是平方！！！！
        gsigma = update_sigma_squarek(samples, ggamma, gmu, gn)
        # 更新bsigma 注意这里是平方！！！！
        bsigma = update_sigma_squarek(samples, bgamma, bmu, bn)

        # 更细您当前参数
        now = [gp, bp, gmu, gsigma, bmu, bsigma]

        # 判断是否继续训练
        # if is_break(cur, now):
        #     break
        # 更新当前参数
        cur = now

        print("Times : \t", times)
        print("Class 1 mean/sigma : \t", gmu, gsigma)
        print("Class 2 mean/sigma : \t", bmu, bsigma)
        print("Class 1/Class 2: \t", bn, gn, bn + gn)
        print("============================")
        times += 1

    return now


# 指定k个Gaussian分布，这里k=2，注意2个Gaussian具有相同的均方差Sigma，均值分别为Mu1，Mu2
def init_data_array(sigma1, sigma2, mu1, mu2, n):
    # 初始化X，为
    x = np.zeros((1, n))

    for i in xrange(0, n):
        if np.random.random(1) > 0.5:
            x[0, i] = np.random.normal() * sigma1 + mu1
        else:
            x[0, i] = np.random.normal() * sigma2 + mu2
    return x


# 指定k个Gaussian分布，这里k=2，注意2个Gaussian具有相同的均方差Sigma，均值分别为Mu1，Mu2
def init_data_list(sigma1, sigma2, mu1, mu2, n):
    # 初始化X，为
    x = []
    for i in xrange(int(n * 0.9)):
        x.append(np.random.normal(mu1, sigma1))
    y = []
    for i in xrange(int(n * 0.1)):
        y.append(np.random.normal(mu2, sigma2))
    samples = []

    print("x size : %d" % (len(x)))
    print("y size : %d" % (len(y)))
    plt.hist(x, 50, normed=True, color=(0.2, 0.7, 0.1))
    plt.hist(y, 50, normed=True, color=(0.6, 0.2, 0.1))
    plt.show()
    samples.extend(x)
    samples.extend(y)

    return samples


# Gauss分布
def generate_data(sigma1, sigma2, mu1, mu2, n):
    x = []
    y = []
    samples = []

    for i in xrange(int(n * 0.6)):
        x.append(random.gauss(mu1, sigma1))
    for i in xrange(int(n * 0.4)):
        y.append(random.gauss(mu2, sigma2))

    samples.extend(x)
    samples.extend(y)
    print("x size : %d" % (len(x)))
    print("y size : %d" % (len(y)))
    plt.hist(x, 50, color=(0.2, 0.9, 0.1))
    plt.hist(y, 50, color=(1, 0.82, 0.82), alpha=0.5)
    plt.hist(samples, 50, color=(0.2, 0.2, 0.8), alpha=0.3)
    plt.show()

    # plt.hist(x, 50, normed=True, color=(0.2, 0.7, 0.1))
    # plt.hist(y, 50, normed=True, color=(0.6, 0.2, 0.1))
    # plt.show()

    return samples


def plot(x):
    plt.hist(x, 50, normed=True, color=(0.2, 0.7, 0.1))
    plt.show()


if __name__ == '__main__':
    # examples = init_data_list(8.0, 8.0, 165, 180, 100000)
    examples = generate_data(8.0, 8.0, 165, 180, 100000)
    calc_em(examples)
    # plot(examples)
