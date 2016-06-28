#!/usr/bin/env python
# coding=utf-8

import re
import numpy as np


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

    def add_document(self, document):
        self.documents.append(document)

    def build_vocabulary(self):
        """
        构建一个词表，词表中的词就是语料中的去重词
        :return:
        """
        discreate_set = set()
        for document in self.documents:
            for word in document.words:
                discreate_set.add(word)
        self.vocabulary = list(discreate_set)

    def plsa(self, nb_topics, max_iter):
        """
        EM
        :param nb_topics:  numbertof topic
        :param max_iter: max number of iterations
        :return:
        """
        print("EM iteration begins....")
        self.build_vocabulary()
        nb_of_documents = len(self.documents)
        vocab_size = len(self.vocabulary)

        # 构建词-文档矩阵
        term_doc_matrix = np.zeros([nb_of_documents, vocab_size], dtype=np.int)
        for d_index, doc in enumerate(self.documents):
            term_count = np.zeros(vocab_size, dtype=np.int)
            for word in doc.words:
                if word in self.vocabulary:
                    w_index = self.vocabulary.index(word)
                    term_count[w_index] += 1
            term_doc_matrix[d_index] = term_count


