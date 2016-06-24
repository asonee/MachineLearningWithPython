# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 15:25:53 2015

@author: TuGuanghui

@E-mail: guanghuitu@gmail.com

实现KNN算法
使用算法需要调节的参数1）K
"""

import numpy as np
import operator

"""
加载简单数据集
@return 数据集
"""
def loadSimpleDataset():
    dataset = [
    [1.0, 1.1, 0],
    [1.0, 1.0, 0],
    [0.0, 0.0, 1],
    [0.0, 0.1, 1]
    ]
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
分类函数
@param dataset
 训练集
@param inX
 需要分类的实例
@param K
 分类时选择的近邻数目
@return 类别
""" 
def classify(dataset, inX, K = 5):
    predictedLabel = None
    dataset = np.array(dataset)
    m = np.shape(dataset)[0]
    sim = np.zeros(m) #存储所有样本与inX之间的相似度
    for i in range(m):
        inst = dataset[i]
        sim[i] = eulidSim(inX, inst[0 : -1])
    sortedSimIndeies = sim.argsort()
    classLabelCount = {} #存放每个类别的样本数
    for i in range(K):
        index = sortedSimIndeies[-(i+1)]
        label = dataset[index][-1]
        classLabelCount[label] = classLabelCount.get(label, 0) + 1 #这样写的话，就可以不用对词典类型进行初始化了
    sortedclassLabelCount = sorted(classLabelCount.iteritems(), key = operator.itemgetter(1), reverse = True )
    predictedLabel = sortedclassLabelCount[0][0]
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
    denom  = np.linalg.norm(inA) *  np.linalg.norm(inB) #分母
    return 0.5 + 0.5 * (num / denom)  #余弦相似度的范围为[-1, 1],放缩到区间[0, 1]，最大最小值放缩方法

"""
对数据集进行最小最大标准化
@param dataset
 数据集
@param minVals
 每个特征对应的最小值
@param maxVals
 每个特征对应的最大值
@return 已标准化好的数据集
""" 
def minMaxStandarization(dataset, minVals, maxVals):
    m = len(dataset)
    dataset  = np.array(dataset)
    staDataset = np.zeros(np.shape(dataset))
    staDataset = dataset[:, 0 : -1] - np.tile(minVals, (m ,1))
    rangeVals = maxVals - minVals
    staDataset = staDataset / np.tile(rangeVals, (m,1))
    staDataset = np.column_stack((staDataset, dataset[:, -1]))
    return staDataset

if __name__ == '__main__':
    #dataset = loadSimpleDataset()
    dataset = loadDataset(".\\datasets\\horseColicTraining2.txt") #K =5时效果最好
    dataset = np.array(dataset)
    minVals = dataset[:, 0:-1].min(0)
    maxVals = dataset[:, 0:-1].max(0)
    dataset = minMaxStandarization(dataset, minVals, maxVals)
    #datasetForTest = dataset
    datasetForTest =  loadDataset(".\\datasets\\horseColicTest2.txt")
    datasetForTest =  minMaxStandarization(datasetForTest, minVals, maxVals)
    correct = 0; wrong = 0
    for inst in datasetForTest:
        if classify(dataset, inst[0:-1], 5) == inst[-1]:
            correct += 1
        else:
            wrong += 1
    print "testing error rate is\t", float(wrong)/(wrong+correct)

