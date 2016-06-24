# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 11:30:25 2015

@author: TuGuanghui

@E-mail: guanghuitu@gmail.com

实现线性回归模型
"""
import matplotlib.pyplot as plt
import numpy as np

"""
加载简单数据集
@return 数据集
"""
def loadSimpleDataset():
    dataset = [[1, 5.56], [2, 5.70], [3, 5.91], [4, 6.40], [5, 6.80], [6, 7.05], [7, 8.90], [8, 8.70], [9, 9.00], [10, 9.05]]
    return dataset

"""
加载数据集
@param fileName
 文件名
@param delimiter
 分隔符
@return 数据集
"""     
def loadDataset(fileName, delimiter = "\t"):
    dataset = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(delimiter)
        floatLine = map(float, curLine)
        dataset.append(floatLine)
    return dataset

"""
训练线性回归模型
@param dataset
 训练数据集
@return 训练好的参数，weight
"""     
def train(dataset):
    datasetMat = np.mat(dataset)
    m, n = np.shape(datasetMat)
    n = n- 1 #让n等于特征的数量，故让n-1
    YMat = datasetMat[:, -1]
    XMat = np.column_stack((np.ones((m,1)), datasetMat[:, 0:-1]))
    XMatTXMatI = (XMat.T*XMat + 0.0001 *np.eye(n+1) ).I #使用伪逆
    #weights = ((XMat.T*XMat).I)*(XMat.T)*YMat
    weights = XMatTXMatI * XMat.T*YMat
    return np.ravel(weights)

"""
利用训练好的线性归回归模型进行预测
@param weights
 训练得到的权重
@param inX
 需要预测的实例
@return 预测值
"""      
def predict(weigths ,inX):
    yHat = 0.0
    for i in range(len(inX)):
        yHat = yHat + inX[i]*weights[i+1]
    yHat = yHat + weights[0]
    return yHat
    
if __name__ == '__main__':
    dataset = loadSimpleDataset()    
    weights = train(dataset)
    yHat = []
    for ins in dataset:
        yHat.append(predict(weights, ins[0 : -1]))
    #图形化显示出来    
    datasetMat = np.mat(dataset)
    plt.plot(datasetMat[:, 0], datasetMat[:, 1], "bo-")        
    plt.plot(datasetMat[:, 0], yHat, "ro-")
    plt.show()
    
    #计算平方误差和
    sumSquareError = 0.0
    for inst in dataset:
        yHat = predict(weights, inst[0 : -1])
        sumSquareError = sumSquareError + (inst[-1] - yHat) * (inst[-1] - yHat)
    print sumSquareError