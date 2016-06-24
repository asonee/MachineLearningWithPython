# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 15:25:53 2015

@author: TuGuanghui

@E-mail: guanghuitu@gmail.com

实现朴素贝叶斯模型，用于文本分类中,注意文本的词权重在这里只能为0-1权重
需要调节的超参数：无
"""
import numpy as np
import math

classes = {} #保存classLabel到classValue的映射,作为全局变量使用，因为有些dataset类标签为字符串类型

"""
加载简单数据集
@return 数据集
"""
def loadSimpleDataset():
    dataset = [
    [1.0, 1.0, 0],
    [1.0, 0.0, 0],
    [1.0, 0.0, 1],
    [1.0, 1.0, 1],
    [1.0, 1.0, 0],
    [0.0, 1.0, 0],
    [0.0, 0.0, 0],
    [0.0, 0.0, 1],
    [0.0, 0.0, 1],
    [0.0, 0.0, 1],
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
训练Naive Bayes，即计算先验概率probOfClass[classIndex]和条件概率probOfWordGivenClass[classIndex][termIndex]   
@param dataset
 训练集
@return 训练好的Naive Bayes
"""
def buildClassifier(dataset):
    m = len(dataset)
    n = len(dataset[0]) -1 
    classesLabels = list(set([inst[-1] for inst in dataset]))
    numClasses = len(classesLabels)
    #初始化classes
    for i in range(numClasses):
        classes[classesLabels[i]] = i #class: classLabel, classValue(classIndex)
    
    probOfClass = np.zeros((numClasses)) #保存先验概率，probOfClass[classIndex]即第classIndex类别的概率
    probOfWordGivenClass = np.zeros((numClasses, n)) #保存term在给定类C中出现的概率，即P(term |C ), probOfWordGivenClass[classIndex][wordIndex]
    #下面计算先验概率probOfClass
    datasetPerCategory = {} #保存classValue:dataset，保存classValue对应的dataset
    for i in range(numClasses):
        datasetPerCategory[i] = [] #初始化datasetPerCategory
    for inst in dataset:#这个循环对数据集进行拆分
        for label in classesLabels:
            if inst[-1] == label:
                datasetPerCategory[classes[label]].append(inst)
    numDocsPerCategory = np.zeros((numClasses))
    for i in range(numClasses):
        numDocsPerCategory[i] = len(datasetPerCategory[i]) #计算每个类别中所含有的样本数
        probOfClass[i] = numDocsPerCategory[i] / float(m) #得到先验概率
    #下面计算条件概率
    for i in range(numClasses):
         for inst in datasetPerCategory[i]:
             for j in range(n):
                 if(inst[j] == 1.0 ):
                     probOfWordGivenClass[i][j] += 1 #probOfWordGivenClass暂时保存某类中出现该term的文档数量
    for i in range(numClasses):
        for j in range(n):
            #probOfWordGivenClass[i][j]  /= float(numDocsPerCategory[i])
            #使用拉普拉斯平滑
            probOfWordGivenClass[i][j]  =  (probOfWordGivenClass[i][j] + 1) / float(numDocsPerCategory[i] + 2)
    model = {"probOfClass":probOfClass, "probOfWordGivenClass":probOfWordGivenClass}
    return model

"""
利用训练好的Naive Bayes分类
@param model
 训练好的Naive Bayes
@param inX
 需要分类的实例
@return 类别
"""     
def classify(model, inX):
    probOfClass = model["probOfClass"]
    probOfWordGivenClass = model["probOfWordGivenClass"]
    nunClasses = len(probOfClass)
    n = len(probOfWordGivenClass[0])
    postProb = np.ones(nunClasses)
    for i in range(nunClasses):
        postProb[i] = math.log(probOfClass[i])
        for j in range(n):
            if inX[j] == 1.0:
                postProb[i] += math.log(probOfWordGivenClass[i][j]) #使用log，连乘变连加，以免小数相乘下溢出
            else:
                postProb[i] += math.log((1 - probOfWordGivenClass[i][j]))
    #将计算得到的后验概率，找到最大的后验概率，并返回对应的标签
    retClassVal = 0.0
    retClassLabel = None
    maxVal = postProb[0]
    for i in range(nunClasses):
        postP = postProb[i]
        if(postP > maxVal):
            maxVal = postP
            retClassVal = i
    for key in classes.keys(): #通过classValue 找到classLabel
        if(classes[key] == retClassVal):
            retClassLabel = key
            break
    return retClassLabel
    
if __name__ == '__main__':
     dataset =  loadSimpleDataset()
     model =  buildClassifier(dataset)
     datasetForTest = dataset
     correct = 0; wrong = 0
     for inst in datasetForTest:
        if classify(model, inst[0:-1]) == inst[-1]:
            correct += 1
        else:
            wrong += 1
            print inst
     print "testing error rate is\t", float(wrong)/(wrong+correct)


