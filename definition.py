# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 10:40:30 2020

@author: Jiajun Liu

BP Neural network homework: functions definition
"""

# load modules

import numpy as np
import matplotlib.pyplot as plt
import time
import random
from dataset_utils import train_set_X, train_set_Y

np.random.seed(1)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def layer_sizes(X , Y):
    """
    参数：
    X - 输入数据集，维度为（输入的数量，训练/测试的数量）
    Y - 标签，维度为（输出的数量，训练/测试的数量）
    
    返回：
    n_x - 输入层的数量
    n_h - 隐藏层的数量
    n_y - 输出层的数量
    """
    
    n_x = X.shape[0] #输入层
    n_h = 60 #隐藏层，硬编码为60
    n_y = Y.shape[0] #输出层
    return (n_x , n_h , n_y)

def initialize_parameters(n_x , n_h , n_y):
    """
    参数：
        n_x - 输入层节点的数量
        n_h - 隐藏层节点的数量
        n_y - 输出层节点的数量
        
    返回：
        parameters - 包含参数的字典：
            W1 - 权重矩阵，维度为（n_h，n_x）
            b1 - 维度为（n_h,1）
            W2 - 权重矩阵，维度为（n_y,n_h）
            b2 - 维度为（n_y,1）
            
    """
    np.random.seed(2) #指定一个随机种子，以便你的输出与我们的一样。
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros(shape = (n_h,1))
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros(shape = (n_y,1))
    
    #use assert to ensure my data format is correct
    assert(W1.shape == (n_h,n_x))
    assert(b1.shape == (n_h,1))
    assert(W2.shape == (n_y,n_h))
    assert(b2.shape == (n_y,1))
    
    parameters = {"W1" : W1,
                  "b1" : b1,
                  "W2" : W2,
                  "b2" : b2
                 }
    
    return parameters

def forward_propagation( X , parameters ):
    """
    参数：
        X - 维度为（n_x,m）的输入数据。
        parameters - 初始化函数（initialize_parameters）的输出
        
    返回：
        A2 - 使用sigmoid()函数计算的第二次激活后的数值
        cache - 包含"Z1"，“A1”，“Z2”和“A2”的字典类型变量
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    #前向传播计算A2
    Z1 = np.dot(W1 , X) + b1
    A1 = sigmoid(Z1) #np.tanh(Z1)
    Z2 = np.dot(W2 , A1) + b2
    A2 = sigmoid(Z2)
    #使用断言确保我的数据格式是正确的
    assert(A2.shape == (1,X.shape[1]))
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return (A2, cache)

def compute_cost(A2,Y,parameters):
    """
    
    参数：
         A2 - 使用sigmoid()函数计算的第二次激活后的数值
         Y - "True"标签向量,维度为（1，数量）
         parameters - 一个包含W1，B1，W2和B2的字典类型的变量

    返回：
         成本 - 交叉熵成本给出方程（13）
    
    """
    
    m = Y.shape[1]
    
    #计算成本
    logprobs = np.multiply(np.log(A2),Y) + np.multiply((1 - Y),np.log(1 - A2))
    cost = - np.sum(logprobs) / m
    cost = float(np.squeeze(cost))
    
    assert(isinstance(cost,float))
    
    return cost

def backward_propagation(parameters , cache , X , Y):
    """
    参数：
     parameters - 包含我们的参数的一个字典类型的变量。
     cache - 包含“Z1”，“A1”，“Z2”和“A2”的字典类型的变量。
     X - 输入数据，维度为（30，数量）
     Y - “True”标签，维度为（1，数量）

    返回：
     grads - 包含W和b的导数一个字典类型的变量。
    """
    
    m = X.shape[1]
    W2 = parameters["W2"]
    
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2 , A1.T)
    db2 = (1 / m) * np.sum(dZ2 , axis = 1 , keepdims = True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), A1*(1-A1))#np.multiply(np.dot(W2.T , dZ2), 1 - np.power(A1 , 2))
    dW1 = (1 / m) * np.dot(dZ1 , X.T)
    db1 = (1 / m) * np.sum(dZ1 , axis = 1 , keepdims = True)
    grads = {"dW1" : dW1,
             "db1" : db1,
             "dW2" : dW2,
             "db2" : db2}
    
    return grads

def update_parameters(parameters,grads,learning_rate=0.5):
    """
    使用上面给出的梯度下降更新规则更新参数

    参数：
     parameters - 包含参数的字典类型的变量。
     grads - 包含导数值的字典类型的变量。
     learning_rate - 学习速率

    返回：
     parameters - 包含更新参数的字典类型的变量。
    """
    
    W1,W2 = parameters["W1"],parameters["W2"]
    b1,b2 = parameters["b1"],parameters["b2"]
    
    dW1,dW2 = grads["dW1"],grads["dW2"]
    db1,db2 = grads["db1"],grads["db2"]
    
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    parameters = {"W1":W1,
                  "b1":b1,
                  "W2":W2,
                  "b2":b2}
    
    return parameters

def nn_model(X,Y,n_h,num_iterations,print_cost=False, eps = 0.01, l=0.5):
    """
    参数：
        X - 数据集,维度为（30，示例数）
        Y - 标签，维度为（1，示例数）
        n_h - 隐藏层的数量
        num_iterations - 梯度下降循环中的迭代次数
        print_cost - 如果为True，则每1000次迭代打印一次成本数值

    返回：
        parameters - 模型学习的参数，它们可以用来进行预测。
    """
    
    np.random.seed(3) #指定随机种子
    n_x = layer_sizes(X,Y)[0]
    n_y = layer_sizes(X,Y)[2]
    
    parameters = initialize_parameters(n_x,n_h,n_y)
    
    costs=[]
    data=open("Sigmoid_Results.txt",'w+') 
    print('================ Sigmoid函数，P={}的BP网络运算结果 ================'.format(n_h)+'\n',file=data)
    for i in range(num_iterations):
        A2 , cache = forward_propagation(X,parameters)
        cost = compute_cost(A2,Y,parameters)
        grads = backward_propagation(parameters,cache,X,Y)
        parameters = update_parameters(parameters,grads,learning_rate=l)
        if print_cost:
            if i%1000 == 0:
                print("第"+str(i)+"次循环，成本为："+str(cost) ,file=data)
                print("第 ", i ," 次循环，成本为：" + str(cost))
                costs.append(cost)
        if cost < eps: 
            print("第 ", i ," 次循环，成本为：" + str(cost))
            print("第"+str(i)+"次循环，成本为："+str(cost),file=data)
            print('递归结束！')
            print('递归结束！'，file=data)
            break
    data.close()
            
    return parameters, costs

def predict(parameters,X):
    """
    
    使用学习的参数，为X中的每个示例预测一个类

    参数：
        parameters - 包含参数的字典类型的变量。
        X - 输入数据（n_x，m）

    返回
        predictions - 我们模型预测的向量（恶性：0 /良性：1）

    """
    
    A2 , cache = forward_propagation(X,parameters)
    predictions = np.round(A2)
    
    return predictions

#tanh()为激活函数模型
def forward_propagation_tanh( X , parameters ):
    """
    参数：
        X - 维度为（n_x,m）的输入数据。
        parameters - 初始化函数（initialize_parameters）的输出
        
    返回：
        A2 - 使用sigmoid()函数计算的第二次激活后的数值
        cache - 包含"Z1"，“A1”，“Z2”和“A2”的字典类型变量
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    #前向传播计算A2
    Z1 = np.dot(W1 , X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2 , A1) + b2
    A2 = sigmoid(Z2)
    #使用断言确保我的数据格式是正确的
    assert(A2.shape == (1,X.shape[1]))
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return (A2, cache)

def backward_propagation_tanh(parameters , cache , X , Y):
    """
    参数：
     parameters - 包含我们的参数的一个字典类型的变量。
     cache - 包含“Z1”，“A1”，“Z2”和“A2”的字典类型的变量。
     X - 输入数据，维度为（2，数量）
     Y - “True”标签，维度为（1，数量）

    返回：
     grads - 包含W和b的导数一个字典类型的变量。
    """
    
    m = X.shape[1]
    W2 = parameters["W2"]
    
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2 , A1.T)
    db2 = (1 / m) * np.sum(dZ2 , axis = 1 , keepdims = True)
    dZ1 = np.multiply(np.dot(W2.T , dZ2), 1 - np.power(A1 , 2))
    dW1 = (1 / m) * np.dot(dZ1 , X.T)
    db1 = (1 / m) * np.sum(dZ1 , axis = 1 , keepdims = True)
    grads = {"dW1" : dW1,
             "db1" : db1,
             "dW2" : dW2,
             "db2" : db2}
    
    return grads

def nn_model_tanh(X,Y,n_h,num_iterations,print_cost=False, eps = 0.01, l=0.5):
    """
    参数：
        X - 数据集,维度为（30，示例数）
        Y - 标签，维度为（1，示例数）
        n_h - 隐藏层的数量
        num_iterations - 梯度下降循环中的迭代次数
        print_cost - 如果为True，则每1000次迭代打印一次成本数值

    返回：
        parameters - 模型学习的参数，它们可以用来进行预测。
    """
    
    np.random.seed(5) #指定随机种子
    n_x = layer_sizes(X,Y)[0]
    n_y = layer_sizes(X,Y)[2]
    
    parameters = initialize_parameters(n_x,n_h,n_y)
    
    costs=[]
    data=open("tanh_Results.txt",'w+') 
    print('================ tanh函数，P={}的BP网络运算结果 ================'.format(n_h)+'\n',file=data)
    for i in range(num_iterations):
        A2 , cache = forward_propagation_tanh(X,parameters)
        cost = compute_cost(A2,Y,parameters)
        grads = backward_propagation_tanh(parameters,cache,X,Y)
        parameters = update_parameters(parameters,grads,learning_rate=l)
        if print_cost:
            if i%1000 == 0:
                print("第"+str(i)+"次循环，成本为："+str(cost) ,file=data)
                print("第 ", i ," 次循环，成本为：" + str(cost))
                costs.append(cost)
        if cost < eps: 
            print("第 ", i ," 次循环，成本为：" + str(cost))
            print("第"+str(i)+"次循环，成本为："+str(cost) ,file=data)
            print('递归结束！')
            print('递归结束！'，file=data)
            break
    data.close()
            
    return parameters, costs

def predict_tanh(parameters,X):
    """
    
    使用学习的参数，为X中的每个示例预测一个类

    参数：
        parameters - 包含参数的字典类型的变量。
        X - 输入数据（n_x，m）

    返回
        predictions - 我们模型预测的向量（红色：0 /蓝色：1）

    """
    
    A2 , cache = forward_propagation_tanh(X,parameters)
    predictions = np.round(A2)
    
    return predictions

#ReLU()为激活函数模型
    
def ReLU(x):    
    return (abs(x) + x) / 2

def ReLU_deriv(x):
    return x>0 #不要使用值传递，应该使用引用传递

def forward_propagation_ReLU( X , parameters ):
    """
    参数：
        X - 维度为（n_x,m）的输入数据。
        parameters - 初始化函数（initialize_parameters）的输出
        
    返回：
        A2 - 使用sigmoid()函数计算的第二次激活后的数值
        cache - 包含"Z1"，“A1”，“Z2”和“A2”的字典类型变量
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    #前向传播计算A2
    Z1 = np.dot(W1 , X) + b1
    A1 = ReLU(Z1)
    Z2 = np.dot(W2 , A1) + b2
    A2 = sigmoid(Z2)
    #使用断言确保我的数据格式是正确的
    assert(A2.shape == (1,X.shape[1]))
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return (A2, cache)

def backward_propagation_ReLU(parameters , cache , X , Y):
    """
    参数：
     parameters - 包含我们的参数的一个字典类型的变量。
     cache - 包含“Z1”，“A1”，“Z2”和“A2”的字典类型的变量。
     X - 输入数据，维度为（2，数量）
     Y - “True”标签，维度为（1，数量）

    返回：
     grads - 包含W和b的导数一个字典类型的变量。
    """
    
    m = X.shape[1]
    W2 = parameters["W2"]
    
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2 , A1.T)
    db2 = (1 / m) * np.sum(dZ2 , axis = 1 , keepdims = True)
    dZ1 = np.multiply(np.dot(W2.T , dZ2), ReLU_deriv(A1))
    dW1 = (1 / m) * np.dot(dZ1 , X.T)
    db1 = (1 / m) * np.sum(dZ1 , axis = 1 , keepdims = True)
    grads = {"dW1" : dW1,
             "db1" : db1,
             "dW2" : dW2,
             "db2" : db2}
    
    return grads

def nn_model_ReLU(X,Y,n_h,num_iterations,print_cost=False, eps = 0.01, l=0.5):
    """
    参数：
        X - 数据集,维度为（2，示例数）
        Y - 标签，维度为（1，示例数）
        n_h - 隐藏层的数量
        num_iterations - 梯度下降循环中的迭代次数
        print_cost - 如果为True，则每1000次迭代打印一次成本数值

    返回：
        parameters - 模型学习的参数，它们可以用来进行预测。
    """
    
    np.random.seed(5) #指定随机种子
    n_x = layer_sizes(X,Y)[0]
    n_y = layer_sizes(X,Y)[2]
    
    parameters = initialize_parameters(n_x,n_h,n_y)
    
    costs=[]            
    data=open("ReLU_Results.txt",'w+') 
    print('================ ReLU函数，P={}的BP网络运算结果 ================'.format(n_h)+'\n',file=data)
    for i in range(num_iterations):
        A2 , cache = forward_propagation_ReLU(X,parameters)
        cost = compute_cost(A2,Y,parameters)
        grads = backward_propagation_ReLU(parameters,cache,X,Y)
        parameters = update_parameters(parameters,grads,learning_rate=l)
        if i%1000 == 0:
            print("第"+str(i)+"次循环，成本为："+str(cost),file=data)
            print("第 ", i ," 次循环，成本为：" + str(cost))
            costs.append(cost)
        if cost < eps: 
            print("第 ", i ," 次循环，成本为：" + str(cost))
            print("第"+str(i)+"次循环，成本为："+str(cost) +'\n',file=data)
            print('递归结束！')
            print('递归结束！'，file=data)
            break
    data.close()
    
    return parameters, costs

def predict_ReLU(parameters,X):
    """
    
    使用学习的参数，为X中的每个示例预测一个类

    参数：
        parameters - 包含参数的字典类型的变量。
        X - 输入数据（n_x，m）

    返回
        predictions - 我们模型预测的向量（红色：0 /蓝色：1）

    """
    
    A2 , cache = forward_propagation_ReLU(X,parameters)
    predictions = np.round(A2)
    
    return predictions

#Leaky ReLU()为激活函数
    
def Leaky_ReLU(x, a=0.01): 
    
    X = x.copy()
    X[X <=0 ] = a * x[X <= 0]    
    
    return X

def Leaky_ReLU_deriv(x, a=0.01):
    
    X = x.copy()
    X[X > 0] = 1
    X[X <= 0] = a
    
    return X #不要使用值传递，应该使用引用传递

def forward_propagation_LeakyReLU( X , parameters ):
    """
    参数：
        X - 维度为（n_x,m）的输入数据。
        parameters - 初始化函数（initialize_parameters）的输出
        
    返回：
        A2 - 使用sigmoid()函数计算的第二次激活后的数值
        cache - 包含"Z1"，“A1”，“Z2”和“A2”的字典类型变量
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    #前向传播计算A2
    Z1 = np.dot(W1 , X) + b1
    A1 = Leaky_ReLU(Z1)
    Z2 = np.dot(W2 , A1) + b2
    A2 = sigmoid(Z2)
    #使用断言确保我的数据格式是正确的
    assert(A2.shape == (1,X.shape[1]))
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return (A2, cache)

def backward_propagation_LeakyReLU(parameters , cache , X , Y):
    """
    参数：
     parameters - 包含我们的参数的一个字典类型的变量。
     cache - 包含“Z1”，“A1”，“Z2”和“A2”的字典类型的变量。
     X - 输入数据，维度为（2，数量）
     Y - “True”标签，维度为（1，数量）

    返回：
     grads - 包含W和b的导数一个字典类型的变量。
    """
    
    m = X.shape[1]
    W2 = parameters["W2"]
    
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2 , A1.T)
    db2 = (1 / m) * np.sum(dZ2 , axis = 1 , keepdims = True)
    dZ1 = np.multiply(np.dot(W2.T , dZ2), Leaky_ReLU_deriv(A1))
    dW1 = (1 / m) * np.dot(dZ1 , X.T)
    db1 = (1 / m) * np.sum(dZ1 , axis = 1 , keepdims = True)
    grads = {"dW1" : dW1,
             "db1" : db1,
             "dW2" : dW2,
             "db2" : db2}
    
    return grads

def nn_model_LeakyReLU(X,Y,n_h,num_iterations,print_cost=False, eps = 0.01, l=0.5):
    """
    参数：
        X - 数据集,维度为（2，示例数）
        Y - 标签，维度为（1，示例数）
        n_h - 隐藏层的数量
        num_iterations - 梯度下降循环中的迭代次数
        print_cost - 如果为True，则每1000次迭代打印一次成本数值

    返回：
        parameters - 模型学习的参数，它们可以用来进行预测。
    """
    
    np.random.seed(5) #指定随机种子
    n_x = layer_sizes(X,Y)[0]
    n_y = layer_sizes(X,Y)[2]
    
    parameters = initialize_parameters(n_x,n_h,n_y)
    
    costs=[]            
    data=open("LeakyReLU_Results.txt",'w+') 
    print('================ Leaky_ReLU函数，P={}的BP网络运算结果 ================'.format(n_h)+'\n',file=data)
    for i in range(num_iterations):
        A2 , cache = forward_propagation_LeakyReLU(X,parameters)
        cost = compute_cost(A2,Y,parameters)
        grads = backward_propagation_LeakyReLU(parameters,cache,X,Y)
        parameters = update_parameters(parameters,grads,learning_rate=l)
        if i%1000 == 0:
            print("第"+str(i)+"次循环，成本为："+str(cost) ,file=data)
            print("第 ", i ," 次循环，成本为：" + str(cost))
            costs.append(cost)
        if cost < eps: 
            print("第 ", i ," 次循环，成本为：" + str(cost))
            print("第"+str(i)+"次循环，成本为："+str(cost) ,file=data)
            print('递归结束！')
            print('递归结束！'，file=data)
            break
    data.close()
            
    return parameters, costs

def predict_LeakyReLU(parameters,X):
    """
    
    使用学习的参数，为X中的每个示例预测一个类

    参数：
        parameters - 包含参数的字典类型的变量。
        X - 输入数据（n_x，m）

    返回
        predictions - 我们模型预测的向量（红色：0 /蓝色：1）

    """
    
    A2 , cache = forward_propagation_LeakyReLU(X,parameters)
    predictions = np.round(A2)
    
    return predictions

#改变隐层节点数
    
def nn_model_LeakyReLU_difN_h(X,Y,n_h,num_iterations,print_cost=False, eps = 0.01, l=0.5):
    """
    参数：
        X - 数据集,维度为（2，示例数）
        Y - 标签，维度为（1，示例数）
        n_h - 隐藏层的数量
        num_iterations - 梯度下降循环中的迭代次数
        print_cost - 如果为True，则每1000次迭代打印一次成本数值

    返回：
        parameters - 模型学习的参数，它们可以用来进行预测。
    """
    
    np.random.seed(5) #指定随机种子
    n_x = layer_sizes(X,Y)[0]
#     n_h = layer_sizes_difN_h(X,Y)[1]
    n_y = layer_sizes(X,Y)[2]
#     print('n_x:{}, n_y:{}'.format(n_x, n_y))
    
    parameters = initialize_parameters(n_x,n_h,n_y)
    
    costs=[]
    end=0
    for i in range(num_iterations):
        A2 , cache = forward_propagation_LeakyReLU(X,parameters)
        cost = compute_cost(A2,Y,parameters)
        grads = backward_propagation_LeakyReLU(parameters,cache,X,Y)
        parameters = update_parameters(parameters,grads,learning_rate=l)
        
        if print_cost:
            if i%1000 == 0:
#                 print("第 ", i ," 次循环，成本为：" + str(cost))
                costs.append(cost)
        if cost <= eps: 
            print("第 ", i ," 次循环，成本为：" + str(cost))
            end = i
            costs.append(cost)
            print('递归结束！')
            print('递归结束！'，file=data)
            break
        elif i >= num_iterations-1:
            print("第 ", i ," 次循环，成本为：" + str(cost))
            end = num_iterations
            print('在规定递归次数内成本未收敛至eps内，请检查梯度是否消失。')
            print('在规定递归次数内成本未收敛至eps内，请检查梯度是否消失。', file=data)
            
            
    return parameters, costs, end

#几种隐层节点数计算函数
    
def Kolmogorov(n_x):
    return int(2*n_x+1)

def Hidnum3(n_x, n_y):
    return int((np.sqrt(0.43*n_x*n_y+0.12*n_y**2+2.54*n_x+0.77*n_y+0.35)+0.51)+0.5)

def Hidnum4(n_s, n_x, n_y, c=1):
    return int(n_s/c/(n_x+n_y)+0.5)

def Hidnum5(n_x, n_y, alp=1):
    return int(np.sqrt(n_x+n_y)+alp+0.5)

def Hidnum6(n_x, n_y):
    return int(np.sqrt(n_x*n_y)+0.5)

def difHiddenLayerNum(HidLayerNum, eps, l=0.5):

    DH_params = []
    DH_costs = []
    times = []
    itrs = []
    tic = time.time()
    data=open("difHidden.txt",'w+') 
    for i in range(len(HidLayerNum)):

        parameters_LeakyReLU, costs_LeakyReLU, End_i = nn_model_LeakyReLU_difN_h(train_set_X, train_set_Y, n_h = HidLayerNum[i], 
                                                                                 num_iterations=200000, print_cost=True, eps=eps, l=l)
        toc = time.time()
        run_time = np.round(toc-tic, 2)
        times.append(run_time)
        tic = toc
        itrs.append(End_i)
        DH_params.append(parameters_LeakyReLU)
        DH_costs.append(costs_LeakyReLU)
        print('================ 隐藏层数为{}的BP网络运算结果 ================'.format(HidLayerNum[i])+'\n',file=data)
        print('eps: {}，学习率：{}，运算时间: {}s，迭代次数: {}'.format(eps,l,run_time,End_i)+'\n',file=data)#.format(datum, eps, run_time, End_i))
        print('W1.shape: {}'.format(parameters_LeakyReLU['W1'].shape),file=data)
        print('W1: ',file=data)
        print(str(parameters_LeakyReLU['W1'])+'\n',file=data)
        print('b1.shape: {}'.format(parameters_LeakyReLU['b1'].shape),file=data)
        print('b1: ',file=data)
        print(str(parameters_LeakyReLU['b1'])+'\n',file=data) 
        print('W2.shape: {}'.format(parameters_LeakyReLU['W2'].shape),file=data)
        print('W2: ',file=data)
        print(str(parameters_LeakyReLU['W2'])+'\n',file=data)
        print('b2.shape: {}'.format(parameters_LeakyReLU['b2'].shape),file=data)
        print('b2: ',file=data)
        print(str(parameters_LeakyReLU['b2']),file=data) 
        print('成本:',file=data)
        print(str(costs_LeakyReLU)+'\n',file=data)
        
    data.close()
    
    return DH_params, DH_costs, times, itrs

#改变学习率
    
def nn_model_LeakyReLU_multi(X,Y,n_h,num_iterations,print_cost=False, eps = 0.01, l=0.5):
    """
    参数：
        X - 数据集,维度为（30，示例数）
        Y - 标签，维度为（1，示例数）
        n_h - 隐藏层的数量
        num_iterations - 梯度下降循环中的迭代次数
        print_cost - 如果为True，则每1000次迭代打印一次成本数值

    返回：
        parameters - 模型学习的参数，它们可以用来进行预测。
    """
    
    np.random.seed(5) #指定随机种子
    n_x = layer_sizes(X,Y)[0]
    n_y = layer_sizes(X,Y)[2]
    
    parameters = initialize_parameters(n_x,n_h,n_y)
    
    costs=[]
    end=0
    for i in range(num_iterations):
        A2 , cache = forward_propagation_LeakyReLU(X,parameters)
        cost = compute_cost(A2,Y,parameters)
        grads = backward_propagation_LeakyReLU(parameters,cache,X,Y)
        parameters = update_parameters(parameters,grads,learning_rate=l)
        
        if print_cost:
            if i%1000 == 0:
                print("第 ", i ," 次循环，成本为：" + str(cost))
                costs.append(cost)
        if cost <= eps: 
            print("第 ", i ," 次循环，成本为：" + str(cost))
            end = i
            print('递归结束！')
            print('递归结束！', file=data)
            break
        elif i >= num_iterations-1:
            print("第 ", i ," 次循环，成本为：" + str(cost))
            end = num_iterations
            print('在规定递归次数内成本未收敛至eps内，请检查梯度是否消失。')
            print('在规定递归次数内成本未收敛至eps内，请检查梯度是否消失。', file=data)
            
            
    return parameters, costs, end

def run_lr_BPNN(lr):
    
    parameters_LeakyReLU, costs_LeakyReLU, end_i = nn_model_LeakyReLU_multi(train_set_X, train_set_Y, n_h = 60, num_iterations=200000, print_cost=True, eps=0.01, l=lr)
    
    return parameters_LeakyReLU, costs_LeakyReLU, end_i

def difLearningRate(LR, eps):

    LR_params = []
    LR_costs = []
    times = []
    itrs = []
    tic = time.time()
    for datum in LR:

        parameters_LeakyReLU, costs_LeakyReLU, End_i = nn_model_LeakyReLU_multi(train_set_X, train_set_Y, n_h = 60, num_iterations=200000, print_cost=True, eps=eps, l=datum)
        toc = time.time()
        run_time = np.round(toc-tic, 2)
        times.append(run_time)
        tic = toc
        itrs.append(End_i)
        print('学习率：{}，eps:{}，运算时间：{}s，迭代次数：{}'.format(datum, eps, run_time, End_i))
        LR_params.append(parameters_LeakyReLU)
        LR_costs.append(costs_LeakyReLU)
    
    return LR_params, LR_costs, times, itrs

def PLOT_diflr(costs,LR, strl, eps):

    plt.figure(figsize=(12, 6),dpi=250)
    plt.rcParams['font.sans-serif']=['SimHei'] 
    plt.rcParams['axes.unicode_minus']=False
    for i in range(len(costs)):
        plt.plot(np.linspace(0,len(costs[i])*1000,len(costs[i])),costs[i],label='学习率:{}'.format(LR[i]))

    plt.legend(loc='upper right')
    plt.title('LeakyReLU:$\epsilon$={}时，不同学习率成本曲线'.format(eps))
    plt.xlabel('迭代次数')
    plt.ylabel('成本')
    plt.savefig('{}.pdf'.format(strl))
#     plt.show()
    return True

#Stochastic Gradient Descent
#有放回的   
def nn_model_SGD_withR(X,Y,n_h, eps = 0.01, l=0.5, epoch=30):
    """
    参数：
        X - 数据集,维度为（30，示例数）
        Y - 标签，维度为（1，示例数）
        n_h - 隐藏层的数量
        num_iterations - 梯度下降循环中的迭代次数
        print_cost - 如果为True，则每1000次迭代打印一次成本数值

    返回：
        parameters - 模型学习的参数，它们可以用来进行预测。
    """
    seed = 0
    np.random.seed(0) #指定随机种子
    lenm = X.shape[1]
    n_x = layer_sizes(X,Y)[0]
    n_y = layer_sizes(X,Y)[2]
    
    parameters = initialize_parameters(n_x,n_h,n_y)
    
    costs=[]
    epoch = epoch#int(num_iterations/lenm)
    for j in range(epoch):
        seed += 1 
        idx = np.random.randint(0,lenm,lenm)## with repalcement
        for i in idx:
            one_X = X[:,i].reshape(-1,1)
            one_Y = Y[:,i].reshape(-1,1)
            A2 , cache = forward_propagation(one_X,parameters)
            cost = compute_cost(A2,one_Y,parameters)
            grads = backward_propagation(parameters,cache,one_X,one_Y)
            parameters = update_parameters(parameters,grads,learning_rate=l)

            if cost < eps: 
                print('{} epoch，第{}次循环，成本为：{},递归结束！'.format(j,i,str(cost)))
                costs.append(cost)
                break
        
    return parameters, costs

#无放回的 
def nn_model_SGD_withOR(X,Y,n_h, eps = 0.01, l=0.5, epoch=30):
    """
    参数：
        X - 数据集,维度为（30，示例数）
        Y - 标签，维度为（1，示例数）
        n_h - 隐藏层的数量
        num_iterations - 梯度下降循环中的迭代次数
        print_cost - 如果为True，则每1000次迭代打印一次成本数值

    返回：
        parameters - 模型学习的参数，它们可以用来进行预测。
    """
    seed = 0
    np.random.seed(seed) #指定随机种子
    lenm = X.shape[1]
    n_x = layer_sizes(X,Y)[0]
    n_y = layer_sizes(X,Y)[2]
    
    parameters = initialize_parameters(n_x,n_h,n_y)
    
    costs=[]
    epoch = epoch
    for j in range(epoch):
        seed += 1 
        idx = np.arange(lenm)
        random.shuffle(idx)  ## without repalcement
        for i in idx:
#             idx = random.sample(range(1, lenm), 1)
            one_X = X[:,i].reshape(-1,1)
            one_Y = Y[:,i].reshape(-1,1)
            A2 , cache = forward_propagation(one_X,parameters)
            cost = compute_cost(A2,one_Y,parameters)
            grads = backward_propagation(parameters,cache,one_X,one_Y)
            parameters = update_parameters(parameters,grads,learning_rate=l)

            if cost < eps: 
                print('{} epoch，第{}次循环，成本为：{},递归结束！'.format(j,i,str(cost)))
                costs.append(cost)
                break
            
    return parameters, costs

#Mini-batch gradient descent

def create_MiniBatch(X,Y,mini_batch_size=64,seed=0):
    '''
    输入：X的维度是（n,m），m是样本数，n是每个样本的特征数
    '''
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []
    #step1：打乱训练集
    permutation = list(np.random.permutation(m))
    #得到打乱后的训练集
    shuffled_X = X[:,permutation]
    shuffled_Y = Y[:,permutation].reshape((1,m))
    #step2：按照batchsize分割训练集
    num_complete_minibatches = m // mini_batch_size #int(m / mini_batch_size)
    for k in range(0,num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size:(k+1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k+1) * mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:,mini_batch_size * num_complete_minibatches:]
        mini_batch_Y = shuffled_Y[:,mini_batch_size * num_complete_minibatches:]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def MiniBatch_SGD(X,Y,n_h,print_cost=False,mini_batch_size=64, eps = 0.01, l=0.5, epoch=200000):
    """
    参数：
        X - 数据集,维度为（30，示例数）
        Y - 标签，维度为（1，示例数）
        n_h - 隐藏层的数量
        num_iterations - 梯度下降循环中的迭代次数
        print_cost - 如果为True，则每1000次迭代打印一次成本数值
8
    返回：
        parameters - 模型学习的参数，它们可以用来进行预测。
    """
    seed = 0
    np.random.seed(seed) #指定随机种子
    n_x = layer_sizes(X,Y)[0]
    n_y = layer_sizes(X,Y)[2]
    
    parameters = initialize_parameters(n_x,n_h,n_y)
    
    costs=[]
    epoch = epoch
    for j in range(epoch):
        seed += 1
        minibatches = create_MiniBatch(X, Y, mini_batch_size=mini_batch_size,seed=seed)
        for minibatch in minibatches:
            minibatch_X, minibatch_Y = minibatch
            A2 , cache = forward_propagation(minibatch_X,parameters)
            cost = compute_cost(A2,minibatch_Y,parameters)
            grads = backward_propagation(parameters,cache,minibatch_X,minibatch_Y)
            parameters = update_parameters(parameters,grads,learning_rate=l)
                            
        if print_cost:
            if j%10 == 0:
                print('{} epoch，成本为：{}'.format(j,str(cost)))
                costs.append(cost)
                
        if cost < eps: 
                print('{} epoch，成本为：{},递归结束！'.format(j,str(cost)))
                costs.append(cost)
                break
            
    return parameters, costs

#Gradient Descent with Momentum
    
def initialize_velocity(parameters):
    """
    将动量初始化为python字典:
        - keys: "dW1", "db1", ..., "dWL", "dbL" 
        - values: 与相应的梯度或参数具有相同形状的全零的numpy array.
    输入:
    parameters -- 包含参数的字典.
        parameters['W' + str(l)] = Wl
        parameters['b' + str(l)] = bl
    
    返回:
    v -- 包含当前动量项的字典.
        v['dW' + str(l)] = velocity of dWl
        v['db' + str(l)] = velocity of dbl
    """
    
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    
    # 初始化动量
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros((parameters['W' + str(l+1)].shape))
        v["db" + str(l+1)] = np.zeros((parameters['b' + str(l+1)].shape))
        
    return v

def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    使用动量更新参数
    
    输入:
    parameters -- 包含参数的字典:
        parameters['W' + str(l)] = Wl
        parameters['b' + str(l)] = bl
    grads -- 包含梯度的字典:
        grads['dW' + str(l)] = dWl
        grads['db' + str(l)] = dbl
    v -- 包含动量项的字典:
        v['dW' + str(l)] = ...
        v['db' + str(l)] = ...
    beta -- 动量超参数，标量
    learning_rate -- 学习率，标量
    
    返回:
    parameters -- 更新后的参数字典
    v -- 更新后的动量项字典
    """

    L = len(parameters) // 2 # number of layers in the neural networks
    
    # Momentum update for each parameter
    for l in range(L):
        # compute velocities
        v["dW" + str(l+1)] = beta * v["dW" + str(l+1)] + (1-beta) * grads['dW'+ str(l+1)]
        v["db" + str(l+1)] = beta * v["db" + str(l+1)] + (1-beta) * grads['db'+ str(l+1)]
        # update parameters
        parameters["W" + str(l+1)] -= learning_rate * v["dW" + str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate * v["db" + str(l+1)]
        
    return parameters, v

def MiniBatch_SGD_withMomentum(X,Y,n_h,print_cost=False,mini_batch_size=64, eps = 0.01, l=0.5, beta = 0.9, epoch=200000):
    """
    参数：
        X - 数据集,维度为（30，示例数）
        Y - 标签，维度为（1，示例数）
        n_h - 隐藏层的数量
        num_iterations - 梯度下降循环中的迭代次数
        print_cost - 如果为True，则每1000次迭代打印一次成本数值
8
    返回：
        parameters - 模型学习的参数，它们可以用来进行预测。
    """
    seed = 0
    np.random.seed(seed) #指定随机种子
    n_x = layer_sizes(X,Y)[0]
    n_y = layer_sizes(X,Y)[2]
    
    parameters = initialize_parameters(n_x,n_h,n_y)
    
    costs=[]
    epoch = epoch
    for j in range(epoch):
        seed += 1
        minibatches = create_MiniBatch(X, Y, mini_batch_size=mini_batch_size,seed=seed)
        for minibatch in minibatches:
            minibatch_X, minibatch_Y = minibatch
            A2 , cache = forward_propagation(minibatch_X,parameters)
            cost = compute_cost(A2,minibatch_Y,parameters)
            grads = backward_propagation(parameters,cache,minibatch_X,minibatch_Y)
            v = initialize_velocity(parameters)
            parameters, v = update_parameters_with_momentum(parameters,grads,v=v,beta=beta,learning_rate=l)
                            
        if print_cost:
            if j%10 == 0:
                print('{} epoch，成本为：{}'.format(j,str(cost)))
                costs.append(cost)
                
        if cost < eps: 
                print('{} epoch，成本为：{},递归结束！'.format(j,str(cost)))
                costs.append(cost)
                break
            
    return parameters, costs
