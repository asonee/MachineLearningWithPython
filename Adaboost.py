# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 21:06:39 2016

@author: TuGuanghui

@E-mail: guanghuitu@gmail.com

Adaboost算法，弱分类器单层决策树或者说决策树桩,算法流程可参考李航的统计学习方法
使用算法需要调节的超参数: 1)弱分类器的数目
算法细节部分：1）当弱分类器的erro >= 0.5, 应将弱分类器的结果进行交换，使得新的error < 0.5(利用树桩当弱分类器，不会出现此问题,因为chooseBestFeatToSplit中交换了左右孩子的节点的1,-1)
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
训练AdaBoost
@param dataset
 训练集
@param M
 需要训练的弱分类器的数目
@return 训练好的模型
"""
def buildClassifier(dataset, M = 50):
    m = np.shape(dataset)[0] #样本数量
    #初始化样本的权值D
    D = np.ones((m,1)) / m
    D = np.ravel(D)
    errorRate = [] #存储每次迭代所产生的最终分类器的error rate
    finalModel = {}
    for i in range(M):
        currentweakModel = buildWeakerClassifer(dataset, D)
        #currentError指的是current Weighted Error
        #print "main", dataset
        currentError = caculateWeightedError(currentweakModel, dataset, D) #计算当前训练出的弱分类器的加权错误率
        #if currentError >= 0.5:
            #当currentError > = 0.5时，调整弱分类器，不可能发生这种情况,因为chooseBestFeatToSplit中交换了左右孩子的节点的1,-1
            #print "currentErrorcurrentErrorcurrentErrorcurrentErrorcurrentErrorcurrentError", currentError            
        currentAlpha = 0.5*math.log((1 - currentError) / currentError) #计算当前弱分类器的alpha值
        finalModel[i] = {"alpha":currentAlpha, "weakModel":currentweakModel}
        #设置提前终止，并记录每次的最终分类器的error rate
        correct = 0; wrong = 0
        for inst in dataset:
            if classify(finalModel, inst[0:-1]) == inst[-1]:
                correct += 1
            else:
                wrong += 1
        print i+1, "\t", float(wrong)/(wrong+correct)
        errorRate.append(float(wrong)/(wrong+correct))
        if wrong == 0: break
        #更新样本集的权值
        currentZ = 0.0
        for i in range(m):
            realClassValue = dataset[i][-1]
            predictedClassValue = weakerClassifierclassify(currentweakModel, dataset[i][0:-1])
            D[i] = D[i] * math.exp(-currentAlpha*realClassValue*predictedClassValue)
            currentZ = currentZ + D[i]
        #规范化样本的权值因子
        for i in range(m):
            D[i] = D[i] / currentZ
    #每次得到的最终模型在训练集上的error rate在图上显示出来，以确定M应该取多少
    weakerClassifierNum = len(finalModel.keys())
    plt.plot(range(1, (weakerClassifierNum+1)), errorRate, "ro-")
    plt.show()
    return finalModel
 
"""
利用训练好的AdaBoost分类
@param model
 训练好的AdaBoost模型
@param inX
 需要分类的实例
@return 类别
"""  
def classify(model, inX):
    fValue = 0.0
    for key in model.keys():
        currentApha =  model[key]["alpha"]
        currentPredictedValue = weakerClassifierclassify(model[key]["weakModel"], inX)
        fValue = fValue + currentApha * currentPredictedValue
    if fValue >= 0:
        return 1
    else:
        return -1

"""
训练弱分类器，弱分类器采用树桩算法
@param dataset
 训练集
@param D
 训练集中每个样本的权重
@return 训练好的弱分类器
"""  
def buildWeakerClassifer(dataset, D):
    return chooseBestFeaturetoSplit(dataset, D)

"""
弱分类器分类函数
@param model
 弱分类器
@param inX
 需要分类的实例
@return 训练好的弱分类器
"""  
def weakerClassifierclassify(model, inX):
    featSplitIndex =  model["spInd"]
    featSplitValue = model["spValue"]
    if inX[featSplitIndex] <= featSplitValue:
        if(isinstance(model["left"], dict)):
            return classify(model["left"], inX)
        else:
            return model["left"]
    else:
        if(isinstance(model["right"], dict)):
            return classify(model["right"], inX)
        else:
 
           return model["right"]
           
"""
根据weightedError评价指标找到最好的切分特征和切分的值
@param dataset
 训练集
@param D
 训练集中每个样本的权重
@return 最好的切分特征和切分值
""" 
def chooseBestFeaturetoSplit(dataset, D):
    m, n = np.shape(dataset)
    n = n -1 #让n等于特征的数量，故让n-1
    lowestWeightedError = np.inf
    bestSplitFeatIndex = 0; bestFeatSplitValue = 0.0; bestLeftLeafNode = 1; bestRightLeafNode = 1
    #对每一个特征的每一个取值的划分得到的MSE进行计算,找到bestFeatIndex，和bestFeatSplitValue
    for featIndex in range(n): #对每一个特征
        featValues = set([ inst[featIndex] for inst in dataset]) #featValues存储该特征不同取值
        if len(featValues) < 10:
            #如果该feature不同的值小于10个，则认为该特征为离散的特征
            numSteps = len(featValues) ##threshVal的个数为len(featValues)+2个，包含了使leftDataset 为空 和rightDataset为空的情况，这样做好
        else:
            #否则认为该特征为离散性特征，thresVal的个数为10+2个，包含了使leftDataset 为空 和rightDataset为空的情况
            numSteps = 10
        rangeMin = min(featValues)
        rangeMax = max(featValues)
        stepSize = (rangeMax - rangeMin) / float(numSteps)
        for j in range(-1, int(numSteps+1)): #对特征的每个取值
            threshVal = rangeMin + float(j) * stepSize # threshVal的个数为numSteps + 2，包含了使leftDataset 为空 和rightDataset为空的情况
            for leftLeafNode in [-1, 1]: #交替左右叶子节点的标签，e.g. left = -1 right = 1 和 left =1 right = -1
                #leftDataset, rightDataset = binSplitDataset(dataset, featIndex, featValue)
                #if (np.shape(leftDataset)[0] < 1) or (np.shape(rightDataset)[0] < 1): continue
                #在当前特征和该特征的切分下得到树桩模型
                tempModel = {}
                tempModel["spInd"] = featIndex
                tempModel["spValue"] = threshVal
                tempModel["left"] = leftLeafNode #直接指定叶子节点的类别，而不是使用多数表决方式
                tempModel["right"] = leftLeafNode * (-1) #直接指定叶子节点的类别， 而不是使用多数表决方式
                tempWeightedError = caculateWeightedError(tempModel, dataset, D)
                if tempWeightedError < lowestWeightedError:
                    lowestWeightedError = tempWeightedError; bestSplitFeatIndex = featIndex; bestFeatSplitValue = threshVal; bestLeftLeafNode = leftLeafNode; bestRightLeafNode = leftLeafNode * (-1)
    retTree = {}
    retTree["spInd"] = bestSplitFeatIndex
    retTree["spValue"] = bestFeatSplitValue
    retTree["left"] = bestLeftLeafNode
    retTree["right"] = bestRightLeafNode
    return retTree

"""
计算加权的error
@param model
 模型
@param dataset
 训练集
@param D
 训练集中每个样本的权重
@return 加权错误
"""
def caculateWeightedError(model, dataset, D):
    weightedError = 0.0
    resultOfClassifySamples = []
    for inst in dataset:
        predictedLabel = weakerClassifierclassify(model, inst[0:-1])
        realLabel = inst[-1]
        if predictedLabel == realLabel:
            resultOfClassifySamples.append(True)
        else:
            resultOfClassifySamples.append(False)
    for i in range(np.shape(dataset)[0]):
        if resultOfClassifySamples[i] == False:
            weightedError = weightedError + D[i]
    return weightedError
    
if __name__ == '__main__':
    #dataset =  loadSimpleDataset()
    dataset = loadDataset(".\\datasets\\horseColicTraining2.txt")
    model =  buildClassifier(dataset, 50)
    #print model
    datasetForTest =  loadDataset("F:\\machine learning\\machinelearninginaction\\Ch07\\horseColicTest2.txt")   
    #datasetForTest = dataset
    correct = 0; wrong = 0
    for inst in datasetForTest:
        if classify(model, inst[0:-1]) == inst[-1]:
            correct += 1
        else:
            wrong += 1
    print "testing error rate is\t", float(wrong)/(wrong+correct)