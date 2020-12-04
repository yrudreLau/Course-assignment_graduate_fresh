# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:40:04 2020

@author: Jiajun Liu

BP Neural network homework: dataset_loads
"""


import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.datasets
from sklearn.model_selection import StratifiedShuffleSplit

breast_cancer = sklearn.datasets.load_breast_cancer()

zh_ch_feature = ['平均半径','平均纹理','平均周长','平均面积','平均平滑度','平均紧致度','平均凹度','平均凹点','平均对称',
                 '平均分形维数','半径误差','纹理误差','周长误差','面积误差','平滑度误差','紧致度误差','凹度误差','凹点误差','对称误差',
                 '分形维数误差','最糟糕的半径','最糟糕的纹理','最糟糕的周长','最糟糕的区域','最糟糕的平滑度','最糟糕的紧致度',
                 '最糟糕的凹度','最差凹点','最差的对称性','最差分形维数']

X = breast_cancer.get('data')
Y = breast_cancer.get('target')

X_ = pd.DataFrame(X, columns=zh_ch_feature)

zh_ch_feature_plus = zh_ch_feature + ['肿瘤']

#归一化
X_norm = pd.DataFrame()
for col in zh_ch_feature:
    ma = X_[col].max()
    mi = X_[col].min()
    X_norm[col] = 0.1 + (X_[col] - mi) / (ma - mi)*(0.9-0.1)
    
X_data = X_norm.values.reshape(X.shape[0],-1)
Y_data = Y.reshape(Y.shape[0],-1)

data = np.hstack((X_data, Y_data))

split = StratifiedShuffleSplit(n_splits = 1,test_size = 0.2,random_state = 42)
for train_index,test_index in split.split(data[:,:-1],data[:,-1]):
    train_set = data[train_index,:]
    test_set = data[test_index,:]

train_set_X = train_set[:,:-1].T
train_set_Y = train_set[:,-1].reshape(-1,1).T
test_set_X = test_set[:,:-1].T
test_set_Y = test_set[:,-1].reshape(-1,1).T