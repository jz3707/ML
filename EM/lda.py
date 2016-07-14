#!/usr/bin/env python
# coding=utf-8

import logging
import logging.config
import ConfigParser
import numpy as np
import random
import codecs
import os
from collections import OrderedDict

# 获取当前路径
path = os.getcwd()

# 导入日志配置文件
logging.config.fileConfig("logging.conf")

# 创建日志对象
logger = logging.getLogger()

# 导入配置文件
conf = ConfigParser.ConfigParser()
conf.read("setting.conf")

# 文件路径
train_file = os.path.join(path, os.path.normpath(conf.get("filepath", "train_file")))
word_id_map_file = os.path.join(path, os.path.normpath(conf.get("filepath", "word_id_map_file")))
theta_file = os.path.join(path, os.path.normpath(conf.get("filepath", "theta_file")))
phi_file = os.path.join(path, os.path.normpath(conf.get("filepath", "phi_file")))
param_file = os.path.join(path, os.path.normpath(conf.get("filepath", "param_file")))
top_N_file = os.path.join(path, os.path.normpath(conf.get("filepath", "top_N_file")))
tassgin_file = os.path.join(path, os.path.normpath(conf.get("filepath", "tassgin_file")))

# 模型初始化参数
K = int(conf.get("model_args", "K"))
alpha = float(conf.get("model_args", "alpha"))
beta = float(conf.get("model_args", "beta"))
iter_max = int(conf.get("model_args", "iter_max"))
top_words_num = int(conf.get("model_args", "top_words_num"))


class Document(object):
    def __init__(self):
        self.words = []
        self.length = 0


class DataPreProcessing(object):
    def __init__(self):
        self.docs_count = 0
        self.words_count = 0
        self.docs = []
        self.word2id = OrderedDict()

    def cache_word_id_map(self):
        with codecs.open(word_id_map_file, 'w', 'utf-8') as f:
            for word, id in self.word2id.items():
                f.write(word + "\t" + str(id) + "\n")


class LDAModel(object):
    def __init__(self, dpre):
        # 获取预处理参数
        self.dpre = dpre

        # 模型参数
        self.K = K
        self.beta = beta
        self.alpha = alpha
        self.iter_max = iter_max
        # 每个topic上取的最多特征词数量
        self.top_words_num = top_words_num
        # θmk，doc m在各个主题上的分布
        self.theta = np.array([[0.0 for _ in xrange(self.K)] for _ in self.dpre.docs_count])
        # φkm，topic k在各个词上的分布
        self.phi = np.array([[0.0 for _ in xrange(self.dpre.words_count)] for _ in xrange(self.K)])

        # 文件变量
        # train_file 分好词的文件
        # 词对应id文件word_id_map_file
        self.word_id_map_file = word_id_map_file
        # 文章-主题分布文件theta_file
        self.theta_file = theta_file
        # 词-主题分布文件phi_file
        self.phi_file = phi_file
        # 每个主题TopN词文件 top_N_file
        self.top_N_file = top_N_file
        # 每个doc分派的主题分拣tassgin_file
        self.tassgin_file = tassgin_file
        # 模型训练选择的参数文件 param_file
        self.param_file = param_file

        # p, 概率向量，采样到某个词的概率，double类型，存储采样的临时变量
        self.sample_prob = np.zeros(self.X)
        # nw, 词word在主题topic上的分布, 维度是[word_count, K]
        self.word_in_topic_count = np.zeros((self.dpre.words_count, self.K), dtype='int')
        # nw_sum，每个topic词的总数, 维度为[K]
        self.topic_words_count = np.zeros(self.K, dtype='int')
        # nd, 每个doc中各个topic的词的总数， 维度为[docs_count, K]
        self.topic_in_doc_count = np.zeros((self.dpre.docs_count, self.K), dtype='int')
        # nd_sum，每个doc中词的总数， 维度[docs_count]
        self.words_in_doc_count = np.zeros(self.dpre.docs_count, dtype='int')

        # Zm,n表示隐含主题分布，隐变量, 就是第m篇doc，第n个词对应的主题。
        # Zm,n 表示α->θm->Zmn->Wmn, β->φk->Wmn
        # θm表示主题分布，α决定主题分布
        # φk表示词分布，β决定词分布
        # LDA过程：
        #   1.从θm中采样出来一个主题Zmn
        #   2.利用Zmn从φk中采样一个词Wmn
        #   PS：其中m是主题编号，n是词编号，
        #       词分布有k个对应K个主题分布。
        # Z的维度[m, n] n是每篇doc的词的总数
        self.Z = np.array([[0 for _ in xrange(self.dpre.docs[m].length)]
                           for m in xrange(self.dpre.docs_count)])

        # 随机分配类型
        for m in xrange(len(self.Z)):
            # doc m中的词总数
            self.words_in_doc_count[m] = self.dpre.docs[m].length

            # 对doc m的每个词随机分配一个主题
            for n in xrange(self.words_in_doc_count[m]):
                # 随机指定主题
                topic = random.randint(0, self.K - 1)
                self.Z[m][n] = topic
                # 词n在topic上的出现的次数
                self.word_in_topic_count[self.words_in_doc_count[m].words[n]][topic] += 1
                # doc m中topic出现的次数
                self.topic_in_doc_count[m][topic] += 1
                # topic上每个词的出现次数
                self.topic_words_count[topic] += 1

    def sampling(self, m, n):
        """
        每次选取概率向量的一个维度，给定其他维度的变量值，采样当前维度的值。
        不断迭代直到收敛。
        统计每个主题Z下出现词n的数量以及每个文档m下出现主题Z的数量
        每一轮计算p(zi|zi-1, d, w)，即排除当前词的主题词
        根据其他所有词的主题分布来估计当前词分配给各个主题的概率

        :param m: doc m
        :param n: word n
        :return:
        """
        topic = self.Z[m][n]
        word = self.dpre.docs[m].words[n]

        # 固定一个维度，给定其他维度的变量值，采样当前维度的值
        self.word_in_topic_count[n][topic] -= 1
        self.topic_in_doc_count[m][topic] -= 1
        self.topic_words_count[topic] -= 1
        self.words_in_doc_count[m] -= 1

        # 先验给定beta
        Vbeta = self.dpre.words_count * self.beta
        # 先验给定alpha
        Kalpha = self.K * self.alpha
        #
















