#!/usr/bin/env python
# coding=utf-8

from random import gammavariate
from random import random


"""
Alpha use a Gamma Distribution.
Samples from a Dirichlet Distribution with alpha.
"""
def Dirichlet(alpha):
    sample = [gammavariate(a, 1) for a in alpha]
    sample = [v / sum(sample) for v in sample]
    return sample

"""
Normalize a vector to be a probablistic representation
"""
def normalize(vec):
    s = sum(vec)
    assert (abs(s) != 0.0)

    for i in range(len(vec)):
        assert (vec[i] >- 0)
        vec[i] = vec[i] * 1.0 / s


"""
Choose a element in @vec according to a specified distribution @pr
"""
def choose(vec, pr):
    assert (len(vec) == len(pr))
    # normalize the distribution
    normalize(vec)
    r = random()
    index = 0
    while r > 0:
        r = r - pr[index]
        index += 1
    return vec[index - 1]

if __name__ == '__main__':
    # test
    print(Dirichlet([1, 1, 1]))
