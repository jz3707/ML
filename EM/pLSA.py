#!/usr/bin/env python
# coding=utf-8

import numpy as np
from utils import normalize


class Document(object):
    def __init__(self, filepath):
        """
        设置文件路径，初始化
        :return:
        """
        self.filepath = filepath

        self.lines = []
        self.words = []

        self.file = open(self.filepath)
        try:
            self.lines = [line for line in self.file]
        finally:
            self.file.close()

    def split(self):
        """
        读取分词结果
        :return:
        """
        for line in self.lines:
            words = line.split(" ")
            for word in words:
                self.words.append(word)


class Corpus(object):
    """
    a collection of document
    """

    def __init__(self):
        self.documents = []
        self.vocabulary = []
        self.nb_of_documents = 0
        self.vocab_size = 0
        self.doc_topic_prob = None
        self.topic_word_prob = None
        self.topic_prob = None
        self.term_doc_matrix = None

    def add_document(self, document):
        self.documents.append(document)

    def build_vocabulary(self):
        """
        构建一个词表，词表中的词就是语料中的去重词
        并且初始化nb_of_documents和vocab_size
        :return:
        """
        discreate_set = set()
        for document in self.documents:
            for word in document.words:
                discreate_set.add(word)
        self.vocabulary = list(discreate_set)
        self.nb_of_documents = len(self.documents)
        self.vocab_size = len(self.vocabulary)

    def generate_term_doc_matrix(self):
        """
         构建词-文档矩阵，统计corpus中单词出现的次数
        :return:
        """
        self.term_doc_matrix = np.zeros([self.nb_of_documents, self.vocab_size], dtype=np.int)
        for d_index, doc in enumerate(self.documents):
            term_count = np.zeros(self.vocab_size, dtype=np.int)
            for word in doc.words:
                if word in self.vocabulary:
                    w_index = self.vocabulary.index(word)
                    term_count[w_index] += 1
            self.term_doc_matrix[d_index] = term_count

    def plsa(self, nb_of_topics, max_iter):
        """
        EM
        :param nb_of_topics:  number of topic
        :param max_iter: max number of iterations
        :return:
        """
        print("EM iteration begins....")
        self.build_vocabulary()
        self.generate_term_doc_matrix()

        # 构造counter arrays
        # p(zk|di) shape : (nb_of_document, nb_of_topics)
        self.doc_topic_prob = np.zeros([self.nb_of_documents, nb_of_topics], dtype=np.float)
        # p(wj|zk) shape: (nb_of_topics, vocab_size)
        self.topic_word_prob = np.zeros([nb_of_topics, self.vocab_size], dtype=np.float)
        # p(zk|di, wj) shape: (nb_of_documents, vocab_size, nb_of_topics)
        self.topic_prob = np.zeros([self.nb_of_documents, self.vocab_size, nb_of_topics], dtype=np.float)

        # Initialize
        print("Initializing ... ")
        # 随机初始
        self.doc_topic_prob = np.random.random(size=(self.nb_of_documents, nb_of_topics))
        for d_index in range(self.nb_of_documents):
            # 归一化每个文档
            normalize(self.doc_topic_prob[d_index])
        self.topic_word_prob = np.random.random(size=(nb_of_topics, self.vocab_size))
        for z in range(nb_of_topics):
            # 归一化每个主题
            normalize(self.topic_word_prob[z])

        # run EM algorithm
        # E-Step:
        #   p(zk|di ,wj) = (p(wj|zk) * p(zk|di)) / (Σl=1,K p(wj|zl) * p(zl|di))
        # M-Step:
        #   p(wj|zk) = Σi n(di, wj) * p(zk|di, wj) / (Σm=1,M Σi n(di, wj)p(zk|di, wj))
        #   p(zk|di) = Σj n(di, wj) * p(zk|di, wj) / (Σk=1,K Σj n(di, wj)p(zk|di, wj))
        for iter in range(max_iter):

            print("Iteration #" + str(iter + 1) + "...")
            print("E step : ")
            for d_index, document in enumerate(self.documents):
                for w_index in range(self.vocab_size):
                    # p(zk|di) * p(wj|zk)
                    # shape : (nb_of_topics), prob是个数组，长度为nb_of_topics
                    prob = self.doc_topic_prob[d_index, :] * self.topic_word_prob[:, w_index]
                    if sum(prob) == 0.0:
                        print("d_index = " + str(d_index) + ", w_index = " + str(w_index))
                        print("self.document_topic_prob[d_index, :] = " + str(self.doc_topic_prob[w_index, :]))
                        print("self.topic_word_prob[:, w_index] = " + str(self.topic_word_prob[:, w_index]))
                        print("topic_prob[d_index][w_index] = " + str(prob))
                        exit(0)
                    else:
                        normalize(prob)
                    self.topic_prob[d_index][w_index] = prob

            print("M step : ")
            # update p(wj|zk)
            for z in range(nb_of_topics):
                for w_index in range(self.vocab_size):
                    numer = 0
                    for d_index in range(self.nb_of_documents):
                        # n(di, wj)
                        count = self.term_doc_matrix[d_index][w_index]
                        # Σi n(di, wj) * p(zk|di, wj)
                        numer += count * self.topic_prob[d_index, w_index, z]
                    self.topic_word_prob[z][w_index] = numer
                normalize(self.topic_word_prob)

            # update p(zk|di)
            for d_index in range(self.nb_of_documents):
                for z in range(nb_of_topics):
                    numer = 0
                    for w_index in range(self.vocab_size):
                        # n(di, wj)
                        count = self.term_doc_matrix[d_index][w_index]
                        numer += count * self.topic_prob[d_index, w_index, z]
                    self.doc_topic_prob[d_index][z] = numer
                normalize(self.doc_topic_prob)
