# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 15:25:53 2015

@author: TuGuanghui

@E-mail: guanghuitu@gmail.com

实现ID3决策树算法
训练树时主要是树的递归思想
"""
import math
import operator

"""
加载简单数据集
@return 数据集
"""
def loadSimpleDataset():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

"""
创建ID3决策树，思想为树的先序遍历
@param dataset
 训练集
@param featLabels
 每个特征的label,或则说每个特征的name
@return 训练好的树模型
"""
def buildClassifier(dataset, featLabels):
    classLabels = [inst[-1] for inst in dataset]
    if classLabels.count(classLabels[0]) == len(classLabels):
        return classLabels[0]   #当数据集中的所有instantces的类别值一样时，直接返回类别
    if len(dataset[0]) == 1:
        return majorityVot(classLabels)  #当数据集中没有特征特征的时候，返回训练集中多数样本的类别
    bestFeatIndex = chooseBestFeatToSplit(dataset)
    bestFeatLabel = featLabels[bestFeatIndex]
    #myTree = {bestFeatLabel:{}} #这句话很神奇，每次递归调用不会重新赋值，而是新增
    myTree = {}
    myTree[bestFeatLabel] = {}
    #接下来利用其余特征建决策树
    bestFeatValues = [inst[bestFeatIndex] for inst in dataset]
    bestFeatUniqueValues = set(bestFeatValues)
    for bestFeatValue in bestFeatUniqueValues:
        subDataset = splitDataset(dataset, bestFeatIndex, bestFeatValue)
        subFeatLables = []
        for featLable in featLabels:
            if featLable != featLabels[bestFeatIndex]:
                subFeatLables.append(featLable) #如果featLabel 不等于 当前最好的特征 featLabel时
        myTree[bestFeatLabel][bestFeatValue] = buildClassifier(subDataset, subFeatLables)
    return myTree

"""
利用训练好的ID3算法分类，思想为树的先序遍历
@param inputTree
 训练好的ID3模型
@param featLabels
 每个特征的label,或则说每个特征的name
@param testInst
 需要分类的实例
@return 类别
"""
def classiy(inputTree, featLabels, testInst):
    rootNode = inputTree.keys()[0]
    rootNodeIndex = featLabels.index(rootNode)
    remainThings = inputTree[rootNode]
    childTree = remainThings[testInst[rootNodeIndex]]
    if isinstance (childTree, dict):
        classLabel = classiy(childTree, featLabels, testInst)
    else:
        classLabel = childTree
    return classLabel

"""
采用信息增益评价标准来选择最佳的切分特征和该特征的最佳切分值
@param dataset
 训练集
@return 最好的切分特征和切分值
""" 
def chooseBestFeatToSplit(dataset):
    #流程： 1、计算每个特征的信息增益 2、找到信息增益最大的特征 3、返回信息增益最大的特征的index
    bestFeatIndex = -1
    originalEntroy = caculateEntropy(dataset)
    bestInfoGain = 0.0
    #计算每个特征划分的数据集的熵
    numFeats = len(dataset[0]) - 1
    for featIndex in range(numFeats):
        featValues = [inst[featIndex] for inst in dataset]
        featUniqValues = set(featValues)
        featEntropy = 0.0;
        #计算每个特征划分的数据的熵，保存在featEntropy
        for featValue in featUniqValues:
            subDataset = splitDataset(dataset, featIndex, featValue)
            prob = float(len(subDataset)) / len(dataset)
            featEntropy = featEntropy + prob * caculateEntropy(subDataset)
        currentInfoGain = originalEntroy - featEntropy
        if currentInfoGain > bestInfoGain:
            bestInfoGain = currentInfoGain
            bestFeatIndex = featIndex
    return bestFeatIndex
    
"""
计算数据集上的熵
@param dataset
 数据集
@return 熵值
"""
def caculateEntropy(dataset):
    #流程：1、分别计算dataset中各个类别的样本数目 2、计算熵 3、返回熵的值
    numInst = len(dataset) # numInst用来存储总共的样本数
    instNumPerClass = {}  #用来存储各个类别的样本数
    #计算各个类别的样本数目
    for inst in dataset:
        currentClassValue = inst[-1]
        if currentClassValue not in instNumPerClass.keys():
            instNumPerClass[currentClassValue] = 0
        instNumPerClass[currentClassValue] = instNumPerClass[currentClassValue] + 1
    #计算熵
    entropy = 0.0;
    for key in instNumPerClass.keys():
        prob = float(instNumPerClass[key])/numInst
        entropy = entropy - prob*math.log(prob,2)
    return entropy

"""
将dataset中去掉featIndex这一列，并且将剩下的数据集featIndex = featValue的instances组成的数据集返回 
@param dataset
 数据集
@param featIndex
 特征的下标
@param featValue
 特征的取值
@return 将数据集featIndex = featValue的instances组成的子数据集返回
"""
def splitDataset(dataset, featIndex, featValue):
    subDataset = []
    for inst in dataset:
        if inst[featIndex] == featValue:
            subInst = []
            subInst = inst[0:featIndex]
            subInst.extend(inst[featIndex+1:])
            subDataset.append(subInst)
    return subDataset
    
"""
多数表决函数
@param dataset
 数据集
@return 数据集中出现最多次的class label
"""
def majorityVot(classLabels):
    classLabelDistribution = {}
    for classLabel in classLabels:
        if classLabel not in classLabelDistribution.keys():
            classLabelDistribution[classLabel] = 0
        classLabelDistribution[classLabel] += 1
    sortedClassLabelDistribution = sorted(classLabelDistribution.iteritems(), key = operator.itemgetter(1), reverse = True)
    return sortedClassLabelDistribution[0][0]
    

if __name__ == '__main__':
    dataset, featLabels =  loadSimpleDataset()
    myTree = buildClassifier(dataset, featLabels)
    print myTree
    print classiy(myTree, featLabels, dataset[2])
   