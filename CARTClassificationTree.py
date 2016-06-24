# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 10:19:18 2015

@author: TuGuanghui

@E-mail: guanghuitu@gmail.com

利用CART做分类树，与CART回归树的不同为采用基尼指数作为准则来划分数据集
代码中让树完全长成，不进行剪枝，包括预剪枝和后剪枝
CART做分类树与ID3, C4.5的不同点：1）采用基尼系数 2）二元切分（二叉树） 3）每次切分完，特征不会被丢弃
"""

import numpy as np
import operator

"""
加载简单数据集
@return 数据集
"""
def loadSimpleDataset():
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
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
训练CART回归树
@param dataset
 训练集
@return 训练好的树模型
"""
def buildClassifier(dataset):
    classLabels = [inst[-1] for inst in dataset]
    #print dataset
    if classLabels.count(classLabels[0]) == len(classLabels):
        #如果该数据集中的所有样本的标签一样，则不进行划分
        return classLabels[0]
    #if np.shape(dataset)[0] < 2:#预剪枝，此部分放在chooseBestFeaturetoSplit()里面最好
        #return majorityVot(dataset)
    bestFeatIndex, bestFeatSplitValue = chooseBestFeaturetoSplit(dataset)
    if bestFeatIndex == None:  #提前终止条件，预剪枝，预剪枝判断全部放在chooseBestFeaturetoSplit中
        return bestFeatSplitValue #将类别放在放在bestFeatSplitValue
    retTree = {}
    retTree["spInd"] = bestFeatIndex
    retTree["spValue"] = bestFeatSplitValue
    leftDatasetMat, rightDatasetMat = binSplitDataset(dataset, bestFeatIndex,  bestFeatSplitValue)
    retTree["left"] = buildClassifier(leftDatasetMat)
    retTree["right"] = buildClassifier(rightDatasetMat)
    return retTree
 
"""
利用训练好的CART分类树分类
@param model
 训练好的CART分类树模型
@param inX
 需要分类的实例
@return 类别
"""     
def classify(treeModel, inX):
    featIndex =  treeModel["spInd"]
    featSplitValue = treeModel["spValue"]
    if inX[featIndex] <= featSplitValue:
        if(isinstance(treeModel["left"], dict)):
            return classify(treeModel["left"], inX)
        else:
            return treeModel["left"]
    else:
        if(isinstance(treeModel["right"], dict)):
            return classify(treeModel["right"], inX)
        else:
            return treeModel["right"]

"""
采用GINI系数评价标准来选择最佳的切分特征和该特征的最佳切分值
@param dataset
 训练集
@return 最好的切分特征和切分值
""" 
def chooseBestFeaturetoSplit(dataset):
    epsilon = 0 #阈值，当小于该阈值时，不进行划分数据集
    m, n = np.shape(dataset)
    n = n -1 #让n等于特征的数量，故让n-1
    orginalGINI = caculateGINI(dataset)
    bestGINI = np.inf
    bestFeatIndex = 0; bestFeatSplitValue = 0.0
    #对每一个特征的每一个取值的划分得到的MSE进行计算,找到bestFeatIndex，和bestFeatSplitValue
    for featIndex in range(n):
        featValues = set([ inst[featIndex] for inst in dataset]) #featValues存储该特征不同取值
        if len(featValues) < 10:
            #离散特征的处理有问题，应该按是否来划分，而不是按数值来划分
            #如果该feature不同的值小于10个，则认为该特征为离散的特征.
            numSteps = len(featValues) #threshVal的个数为len(featValues)个
        else:
            #否则认为该特征为离散性特征，thresVal的个数为10个
            numSteps = 10
        rangeMin = min(featValues)
        rangeMax = max(featValues)
        stepSize = (rangeMax - rangeMin) / float(numSteps)
        for j in range(0, int(numSteps)): #对特征的每个取值
            threshVal = rangeMin + float(j) * stepSize # threshVal的个数为numSteps
            leftDataset, rightDataset = binSplitDataset(dataset, featIndex, threshVal)
            newGINI = (len(leftDataset)/float(m)) * caculateGINI(leftDataset) + (len(rightDataset)/float(m))*caculateGINI(rightDataset)
            if newGINI < bestGINI:
                bestGINI = newGINI; bestFeatIndex = featIndex; bestFeatSplitValue = threshVal
    if(orginalGINI - bestGINI == epsilon):
        #此处是为了防止噪声数据，导致gini系数没有变，例如当前dataset为[1,1,0,-1],[1,1,0,1]，不管怎样都不能把它们分开
        #若不这样的话，会导致leftDataset为空,这个并不是剪枝
        return None, majorityVot(dataset) #如果GINI系数没有变化，则不进行划分
    return bestFeatIndex, bestFeatSplitValue

"""
计算基尼系数
@param dataset
 数据集
@return 基尼系数值
"""
def caculateGINI(dataset):
    numInst = len(dataset) # numInst用来存储总共的样本数
    if numInst == 0.0:
        return 0.0 #对于空数据集返回基尼系数为0
    instNumPerClass = {}  #用来存储各个类别的样本数
    #计算各个类别的样本数目
    for inst in dataset:
        currentClassValue = inst[-1]
        if currentClassValue not in instNumPerClass.keys():
            instNumPerClass[currentClassValue] = 0
        instNumPerClass[currentClassValue] = instNumPerClass[currentClassValue] + 1
    #计算GINI系数
    gini = 0.0;
    for key in instNumPerClass.keys():
        prob = float(instNumPerClass[key])/numInst
        gini = gini + prob * prob
    gini = 1 - gini
    return gini

"""
多数表决函数
@param dataset
 数据集
@return 数据集中出现最多次的class label
"""
def majorityVot(dataset):
    classLabels =  [inst[-1] for inst in dataset]
    classLabelDistribution = {}
    for classLabel in classLabels:
        if classLabel not in classLabelDistribution.keys():
            classLabelDistribution[classLabel] = 0
        classLabelDistribution[classLabel] += 1
    sortedClassLabelDistribution = sorted(classLabelDistribution.iteritems(), key = operator.itemgetter(1), reverse = True)
    return sortedClassLabelDistribution[0][0]

"""
根据给定特征和该特征的某个取值对数据集进行二元切分
@param dataset
 数据集
@param featIndex
 特征的下标
@param threshVal
 阀值
@return 切分好的两个子数据集
""" 
def binSplitDataset(dataset, featIndex, threshVal):
    leftDataset = []
    rightDataset = []
    leftIndex = np.nonzero([inst[featIndex] <= threshVal for inst in dataset])[0]
    rightIndex = np.nonzero([inst[featIndex] > threshVal for inst in dataset] )[0]
    for i in leftIndex:
        leftDataset.append(dataset[i])
    for i in rightIndex:
        rightDataset.append(dataset[i])
    return leftDataset, rightDataset

if __name__ == '__main__':
    #dataset = loadSimpleDataset()
    dataset = loadDataset(".\\datasets\\horseColicTraining2.txt") 
    myTree =  buildClassifier(dataset)
    #print myTree
    correct = 0; wrong = 0
    for inst in dataset:
        if classify(myTree, inst[0:-1]) == inst[-1]:
            correct += 1
        else:
            wrong += 1
    print "Accuracy: ", float(correct) / (wrong + correct)
    