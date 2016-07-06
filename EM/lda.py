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
        self.top_words_num = top_words_num

        # 文件变量
        # train_file 分好词的文件
        # 词对应id文件word_id_map_file
        self.word_id_map_file = word_id_map_file
        # 文章-主题分布文件theta_file
        self.theta_file =theta_file
        # 词-主题分布文件phi_file
        self.phi_file = phi_file
        #





