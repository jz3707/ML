#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function

from argparse import ArgumentParser


import pandas as pd
import numpy as np
import xgboost as xgb
# using the most importance feature
# filtered features is 246 ones.
index = pd.read_csv("./xgb.feat_importance")
index = index['feature']
# 从readme.txt中下载
df_data = pd.read_csv('./HFT_XY_unselected.csv')
col_names_x = df_data.columns[1:333]
x_all = np.asarray(df_data[index])
col_names_y = df_data.columns[333]
y_all = np.asarray(df_data[col_names_y])
print(x_all.shape)
print(x_all.shape)


train_x = x_all[0:x_all.shape[0]*7/10]
train_y = y_all[0:y_all.shape[0]*7/10]
test_x = x_all[x_all.shape[0]*7/10:]
test_y = y_all[y_all.shape[0]*7/10:]
print(train_y.shape)
print(train_x.shape)
print(test_x.shape)
print(test_y.shape)

# Now let's Learn some models...
# benchmark model is OLS

from sklearn import linear_model
from sklearn.metrics import r2_score

ols = linear_model.LinearRegression()
pred_ols = ols.fit(train_x, train_y).predict(test_x)
print(r2_score(test_y, pred_ols))








