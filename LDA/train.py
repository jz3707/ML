#!/usr/bin/env python
# coding=utf-8

import cPickle, getopt, sys, time, re
import datetime, os
import scipy.io
import numpy as np
import optparse


def parse_args():
    parser = optparse.OptionParser()

    parser.set_default(
        # parameter set 1
        input_directory=None,
        output_directory=None,

        # parameter set 2
        training_iterations=-1,
        snapshot_interval=10,
        nb_topics=-1,

        # parameter set 3
        alpha_alpha=-1,
        alpha_beta=-1,

        # parameter set 4
        inference_mode=0, )

    # parameter set 1
    parser.add_option("--input_directory", type="string", dest="input_directory",
                      help="input directory [None]")
    parser.add_option("--output_directory", type="string", dest="output_directory",
                      help="output directory [None]")

    # parameter set 2
    parser.add_option("--nb_topics", type="int", dest="nb_topics",
                      help="number of  topic [-1]")
    parser.add_option("--training_iterations", type="int", dest="training_iterations",
                      help="training_iterations [-1]")
    parser.add_option("--snapshot_interval", type="int", dest="snapshot_interval",
                      help="snapshot_interval [10]")

    # parameter set 3
    parser.add_option("--alpha_alpha", type="int", dest="alpha_alpha",
                      help="hyper-parameter for Dirichlet distribution of topics. [1.0/nb_topics]")
    parser.add_option("--alpha_beta", type="int", dest="alpha_beta",
                      help="hyper-parameter for Dirichlet distribution of vocabulary. [1.0/nb_types]")

    # parameter set 4
    parser.add_option("--inference_mode", type="int", dest="inference_mode",
                      help="inference_mode [" +
                           "0 (default) : hybrid inference" +
                           "1 : monte carlo, " +
                           "2 : variational bayes" +
                           "]")

    (options, args) = parser.parse_args()
    return options


def main():
    options = parse_args()

    # parameter set 2
    assert (options.nb_topics > 0)
    nb_topics = options.nb_topics
    assert (options.training_iterations > 0)
    training_iterations = options.training_iterations
    assert (options.snapshot_interval > 0)
    snapshot_interval = options.snapshot_interval

    # parameter set 4
    inference_mode = options.inference_mode

    # parameter set 1
    assert (options.input_directionary is not None)
    assert (options.output_directionary is not None)

    input_directionary = options.input_directionary
    input_directionary = input_directionary.rstrip("/")
    corpus_name = os.path.basename(input_directionary)

    output_directionary = options.output_directionary

    if not os.path.exists(output_directionary):
        os.mkdir(output_directionary)
    output_directionary = os.path.join(output_directionary, corpus_name)

    if not os.path.exists(output_directionary):
        os.mkdir(output_directionary)

    # Document
    train_docs_path = os.path.join(input_directionary, 'train.dat')
    input_doc_stream = open(train_docs_path, 'r')
    train_docs = []
    for line in input_doc_stream:
        train_docs.append(line.strip().lower())
    print('successfully load all training docs from %s' % (os.path.abspath(train_docs_path)))

    # Vocabulary
    vocabluary_path = os.path.join(input_directionary, 'voc.dat')
    input_voc_stream = open(vocabluary_path, 'r')
    vocab = []
    for line in input_voc_stream:
        vocab.append(line.strip().lower().split()[0])
    vocab = list(set(vocab))
    print('successfully load all the words from %s ...' % (os.path.abspath(vocabluary_path)))

    # parameter set 3
    alpha_alpha = 1.0 / nb_topics
    if options.alpha_alpha > 0:
        alpha_alpha = options.alpha_alpha
    alpah_beta = options.alpha_beta
    if alpah_beta <= 0:
        alpha_beta = 1.0 / len(vocab)



















