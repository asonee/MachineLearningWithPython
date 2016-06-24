# -*- coding: utf-8 -*-
"""
Created on Wed Nov 4 18:16:25 2015

@author: TuGuanghui

@E-mail: guanghuitu@gmail.com

实现推拉质心向量算法
使用该算法需要调节的参数1）学习率 2）最大迭代次数
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
    dataset = [[0.0,1],[1.0,1],[2.0,1],[3.0,-1],[4.0,-1],[5.0,-1],[6.0,1],[7.0,1],[8.0,1],[9.0,-1]]
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
训练推拉算法
@param dataset
 训练集
@param M
 最大迭代次数
@param alpha
 学习率、步长
@return 训练好的模型
""" 
def buildClassifier(dataset, M = 50, alpha = 0.001):
    centroids = []
    centroids = computeCentroids(dataset)
    correct = 0; wrong = 0
    for inst in dataset:
        if classify(centroids, inst[0:-1]) == inst[-1]:
            correct += 1
        else:
            wrong += 1
    print "original traning error rate is\t", float(wrong)/(wrong+correct)
    #对得到的原始质心进行推拉操作,使用miniBatch的方式进行推拉操作
    errorRate = [float(wrong)/(wrong+correct)] #用来保存M次的errorRate，用于判断训练是否收敛
    for m in range(M):
        misclassifiedSamples = {}
        for key in centroids.keys():
            #初始化misclassifiedSamples
            misclassifiedSamples[key] = {}
            misclassifiedSamples[key]["FN"] = []
            misclassifiedSamples[key]["FP"] = []
        for i in range(len(dataset)):
            realLabel = dataset[i][-1]
            predictedLable = classify(centroids, dataset[i][0 : -1])
            if(realLabel != predictedLable):
                misclassifiedSamples[realLabel]["FN"].append(i) #FN存储为本来为realLabel的，但被误分为其他类别
                misclassifiedSamples[predictedLable]["FP"].append(i) #FP存储为不是predictedLable，但被误分到predictedLable
        #接下来对centroids进行更新
        for key in centroids.keys():
            FN = misclassifiedSamples[key]["FN"]
            FP = misclassifiedSamples[key]["FP"]
            FNTotal = sumAllSamples(dataset, FN) #存储FN中所有样本的和
            FPTotal = sumAllSamples(dataset, FP) #存储FP中所有样本的和
            centroids[key] = vecNormalization( list(np.array(centroids[key]) + alpha * np.array(FNTotal) - alpha * np.array(FPTotal))) #对质心进行推拉，并且更新质心
        #对得到的新的centroids
        correct = 0; wrong = 0
        for inst in dataset:
            if classify(centroids, inst[0:-1]) == inst[-1]:
                correct += 1
            else:
                wrong += 1
        print "traning error rate is\t", float(wrong)/(wrong+correct)
        errorRate.append(float(wrong)/(wrong+correct))
    plt.plot(range(0, (M+1)), errorRate, "ro-")
    plt.show()
    return centroids

"""
给定下标所指向的样本全部加起来，e.g [1,2] + [3, 4] = [4, 6]
@param dataset
 数据集
@param indies
 给定样本的下标
@return 样本和
""" 
def sumAllSamples(dataset, indies):
        sum = np.zeros(len(dataset[0]) - 1)
        for index in indies:
            sum = sum + np.array(dataset[index][0 : -1])
        return list(sum)
 
"""
利用训练好的推拉算法进行分类
@param model
 训练好的推拉算法
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
    return 1.0 / (1.0 + np.linalg.norm(inA - inB)) #使相似度在区间(0, 1]

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
对向量进行规范化，使向量的摸长为1
@param inX
 实例
@return 模长为1的向量
""" 
def vecNormalization(inX):
    n = len(inX)
    inX = np.array(inX)
    denom = np.linalg.norm(inX) #分母
    if denom == 0.0: #如果向量的摸长为1，则不进行规范化
        return inX
    for i in range(n):
        inX[i] = inX[i] / float(denom)
    return list(inX)

"""
整个数据集进行规范化，使向量的模长为1
@param dataset
 数据集
@return 已规范化的数据集
"""
def datasetNormalization(dataset):
    for i in range(len(dataset)):
        inst = dataset[i]
        norInst = vecNormalization(inst[0 : -1])
        norInst = list(norInst)
        norInst.append(inst[-1])
        dataset[i] = norInst
    return dataset

"""
计算数据集中各个类别的质心
@param dataset
 数据集
@return 各个类别的质心
"""   
def computeCentroids(dataset):
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
        centroids[key] = vecNormalization(centroid) #规范化质心
    return centroids
    
if __name__ == '__main__':
    #dataset =  loadSimpleDataset()
    dataset = loadDataset(".\\datasets\\horseColicTraining2.txt") #最大跌代次数为：23 学习率为： 0.0003
    #将数据进行规范化
    dataset = datasetNormalization(dataset)
    model = buildClassifier(dataset, 23, 0.0003)
    
    #datasetForTest = dataset
    datasetForTest =  loadDataset("F:\\machine learning\\machinelearninginaction\\Ch07\\horseColicTest2.txt")   
    datasetForTest = datasetNormalization(datasetForTest)
    correct = 0; wrong = 0
    for inst in datasetForTest:
        if classify(model, inst[0:-1]) == inst[-1]:
            correct += 1
        else:
            wrong += 1
    print "testing error rate is\t", float(wrong)/(wrong+correct)
