# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 20:18:29 2015

@author: TuGuanghui

@E-mail: guanghuitu@gmail.com

实现PCA算法
"""
import numpy as np
import matplotlib.pyplot as plt

"""
加载简单数据集
@return 数据集
"""
def loadSimpleDataset():
    dataset = [[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0], [2.3, 2.7], [2, 1.6], [1, 1.1], [1.5, 1.6], [1.1, 0.9]]
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
PCA方法
@param dataset
 数据集
@param topFeatNum
 需要保留的特征的个数
@return 投影后的数据
"""
def PCA(dataset, topFeatNum = 2):
    #步骤如下
    #1、首先将数据零均值标准化 
    #2、计算数据的协方差矩阵 
    #3、对协方差矩阵进行特征值分解 
    #4、选择xx个特征向量作为新的基 
    #5、将原始数据投影到新的基上
    datasetMat = np.mat(dataset)
    meanValues = np.mean(datasetMat, axis = 0)
    stds = np.std(datasetMat, axis = 0)
    adjustedDatasetMat = datasetMat - meanValues
    adjustedDatasetMat = adjustedDatasetMat / stds
    plt.plot(adjustedDatasetMat[:, 0], adjustedDatasetMat[:, 1], "r^")
    plt.show()
    covMat = np.cov(adjustedDatasetMat, rowvar = 0)
    #covMat = (adjustedDatasetMat.T * adjustedDatasetMat) / datasetMat.shape[0] #由于均值为0，所以可以这样做
    eigenVals, eigenVecs = np.linalg.eig(np.mat(covMat))
    draw(eigenVals) #通过看图像选择需要保留的特征的数目
    eigenValsIndex = np.argsort(eigenVals) #将eigenVals从小到大进行排序，并将下标返回
    eigenValsIndex = eigenValsIndex[: -(topFeatNum+1) : -1] #返回eigenVals最高的topFeatNum个值得下标
    eigenVecs = eigenVecs[:, eigenValsIndex] #仅仅选取topFeatNum个特征向量并按照eigenValues进行排序
    transformedDatasetMat = adjustedDatasetMat * eigenVecs
    
    return transformedDatasetMat

"""
画图，此图横坐标为前n个主成分的数目，纵坐标为占总的方差的比例，以此来确定应该要保留的特征数目
@param eigenVals
 特征值
"""
def draw(eigenVals):
    eigenVals.sort() #将eigenVals从小到大进行排序
    lens = eigenVals.size
    X = range(1, (lens+1))
    Y = np.zeros(lens)
    Y[0] = eigenVals[-1]
    for i in range(2, (lens+1)):
        Y[i-1] = Y[i-2] + eigenVals[-i]
    sumVars = np.sum(eigenVals)
    for i in range(lens):
        Y[i] = Y[i] / sumVars
    plt.plot(X, Y, "ro-")
    plt.show()

if __name__ == '__main__':
    dataset = loadDataset(".\\datasets\\testSet.txt")
    #dataset = loadSimpleDataset()
    transformedDatasetMat = PCA(dataset, 2)
    #可视化转换后的数据
    plt.plot(transformedDatasetMat[:, 0], transformedDatasetMat[:, 1],"go")
    plt.show()
