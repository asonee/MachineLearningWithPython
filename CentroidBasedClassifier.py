# -*- coding: utf-8 -*-
"""
Created on Tue Nov 3 10:16:25 2015

@author: TuGuanghui

@E-mail: guanghuitu@gmail.com

实现质心向量算法
"""
import numpy as np
import matplotlib.pylab as plt
import math

"""
加载简单数据集
@return 数据集
"""
def loadSimpleDataset():
    """
    dataset = [[1, 2.1, 1.0],
               [2, 1.1, 1.0],
               [1.3, 1, -1.0],
               [1, 1, -1.0],
               [2, 1, 1.0]
               ]
    """
    dataset = [[0,1],[1,1],[2,1],[3,-1],[4,-1],[5,-1],[6,1],[7,1],[8,1],[9,-1]]
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
训练Centroid Based Classifier
@param dataset
 训练集
@return 训练好的模型
"""   
def buildClassifier(dataset):
    m = len(dataset) #样本的数量
    n = len(dataset[0]) - 1 #特征的数量
    centroids = {}
    classLabels = list(set([inst[-1] for inst in dataset])) #得到所有类别的标签
    #扫描数据集，得到仅含单个类别的数据集
    datasetPerClass = {} 
    for i in range(len(classLabels)):
        datasetPerClass[classLabels[i]] = [] #初始化datasetPerClass
        centroids[classLabels[i]] = [] #初始化质心向量
    for inst in dataset:
        for i in range(len(classLabels)):
            if inst[-1] == classLabels[i]:
                datasetPerClass[classLabels[i]].append(inst)
                break
    for key in datasetPerClass.keys():
        currentDataset = datasetPerClass[key]
        #以下计算对应类别的数据集的质心
        centroid = []
        total = [] #total存储currentDataset中每一列特征的和
        for i in range(n):
            total.append(0.0)
        for inst in currentDataset: #得到每一列特征的和
            for featIndex in range(n):
                total[featIndex]  += inst[featIndex]
        for featIndex in range(n):
            centroid.append(total[featIndex] / float(m))
        centroids[key] = centroid
    correct = 0; wrong = 0
    for inst in dataset:
        if classify(centroids, inst[0:-1]) == inst[-1]:
            correct += 1
        else:
            wrong += 1
    print "original traning error rate is\t", float(wrong)/(wrong+correct)
    return centroids

"""
利用训练好的CBC分类
@param model
 训练好的CBC模型
@param inX
 需要分类的实例
@return 类别
"""     
def classify(model, inX):
    maxSim = -1
    predictedLabel = None
    for key in model:
        sim = eulidSim(model[key], inX)
        if(sim > maxSim):
            maxSim = sim
            predictedLabel = key
    return predictedLabel

"""
两个实例之间的欧几里得相似度
@param inA
 实例A
@param inB
 实例B
@return 欧几里得相似度
"""     
def eulidSim(inA, inB):
    inA = np.array(inA)
    inB = np.array(inB)
    eulidSim =  np.sqrt(np.sum(np.power(inA - inB, 2)))
    #eulidSim = np.np.linalg.norm(inA - inB)
    normEulidSim = 1.0 / (1.0 + eulidSim) #使相似度在区间(0, 1]
    return normEulidSim 

"""
两个实例之间的cosine相似度
@param inA
 实例A
@param inB
 实例B
@return cosine相似度
"""       
def cosSim(inA, inB):
    inA = np.array(inA)
    inB = np.array(inB)
    num = float(inA.dot(inB))
    denom  = np.linalg.norm(inA) *  np.linalg.norm(inB)
    return 0.5 + 0.5 * (num / denom)  #余弦相似度的范围为[-1, 1],放缩到区间[0, 1]，最大最小值放缩方法
    
if __name__ == '__main__':
    #dataset =  loadSimpleDataset()
    dataset = loadDataset("F:\\machine learning\\machinelearninginaction\\Ch07\\horseColicTraining2.txt")
    model = buildClassifier(dataset)
    #datasetForTest = dataset
    datasetForTest =  loadDataset("F:\\machine learning\\machinelearninginaction\\Ch07\\horseColicTest2.txt")   
    correct = 0; wrong = 0
    for inst in datasetForTest:
        if classify(model, inst[0:-1]) == inst[-1]:
            correct += 1
        else:
            wrong += 1
    print "testing error rate is\t", float(wrong)/(wrong+correct)
    

