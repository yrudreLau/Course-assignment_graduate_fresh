# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:50:08 2020

@author: Jiajun Liu

BP Neural network homework
"""
## 谨慎运行
import numpy as np
from definition import nn_model, predict
from dataset_utils import train_set_X, train_set_Y, test_set_X, test_set_Y

parameters, costs = nn_model(train_set_X, train_set_Y, n_h = 60, num_iterations=200000, 
                             print_cost=True, eps=0.01, l=0.5)

predictions_sigmoid_train = predict(parameters, train_set_X)
print('训练集准确率: %d' % float((np.dot(np.squeeze(train_set_Y), predictions_sigmoid_train.T) + np.dot(1 - np.squeeze(train_set_Y), 1 - predictions_sigmoid_train.T)) / float(train_set_Y.size) * 100) + '%')

predictions_sigmoid_test = predict(parameters, test_set_X)
print('测试集准确率: %d' % float((np.dot(np.squeeze(test_set_Y), predictions_sigmoid_test.T) + np.dot(1 - np.squeeze(test_set_Y), 1 - predictions_sigmoid_test.T)) / float(test_set_Y.size) * 100) + '%')
