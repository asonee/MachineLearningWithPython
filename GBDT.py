# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 21:06:39 2016

@author: TuGuanghui

@E-mail: guanghuitu@gmail.com

梯度提升树，弱分类器选用树桩，损失函数选用平方损失函数
使用算法需要调节的超参数: 1)弱分类器的数目
"""
import matplotlib.pyplot as plt
import numpy as np
import copy

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
训练GBDT
@param dataset
 训练集
@param M
 需要训练的弱分类器的数目
@return 训练好的模型
"""    
def train(dataset, M = 30):
    combinedModel = {} #组合的最终模型
    sumSquareErrors = [] #用于判断模型是否收敛
    for i in range(M):
        #计算残差
        residuals =  []
        if i == 0:
            #第一次迭代时，残差就是样本的y值
            residuals = [inst[-1] for inst in dataset]
        else:
            for inst in dataset:
                residual = inst[-1] - predict(combinedModel, inst[0 : -1])
                residuals.append(residual)
        combinedModel[i] = trainRegressionStump(dataset, residuals)
        #计算当前组合的模型下，预测数据的平方误差和
        sumSquareError = 0.0
        for inst in dataset:
            yHat = predict(combinedModel, inst[0 : -1])
            sumSquareError = sumSquareError + (inst[-1] - yHat) * (inst[-1] - yHat)
        sumSquareErrors.append(sumSquareError)
    plt.plot(range(1, M+1), sumSquareErrors, "ro-")
    plt.show()
    return combinedModel

"""
利用训练好的GBDT预测
@param model
 训练好的GBDT
@param inX
 需要预测的实例
@return 预测值
"""     
def predict(model ,inX):
    yHat = 0.0
    for key in model.keys():
        stump = model[key]
        yHat = yHat + regressionStumpPredict(stump, inX)
    return yHat

"""
训练回归树桩
@param dataset
 训练集
@param residuals
 需要拟合的残差
@return 训练好的回归树桩
"""
def trainRegressionStump(dataset, residuals):
    newDataset = copy.deepcopy(dataset)
    for i in range(len(dataset)):
        newDataset[i][-1] = residuals[i]
    return chooseBestFeaturetoSplit(newDataset)
    
"""
利用回归树桩进行预测 
@param model
 训练好的回归树桩
@param inX
 需要预测的实例
@return 预测值
""" 
def regressionStumpPredict(model, inX):
     featIndex = model["spInd"] 
     featValue = model["spValue"]
     if inX[featIndex] < featValue:
         return model["left"]
     else:
         return model["right"]

"""
根据最小平方差的和评价指标找到最好的切分特征和切分的值
@param dataset
 训练集
@return 最好的切分特征和切分值
""" 
def chooseBestFeaturetoSplit(dataset):
    m, n = np.shape(dataset)
    n = n -1 #让n等于特征的数量，故让n-1
    lowestSquareError = np.inf
    bestSplitFeatIndex = 0; bestFeatSplitValue = 0.0; bestLeftLeafNode = 0.0; bestRightLeafNode = 0.0
    #对每一个特征的每一个取值的划分得到的MSE进行计算,找到bestFeatIndex，和bestFeatSplitValue
    for featIndex in range(n): #对每一个特征
        for featValue in set([ inst[featIndex] for inst in dataset]): #对特征的每个取值
             leftDataset, rightDataset = binSplitDataset(dataset, featIndex, featValue)
             if len(leftDataset) == 0 or len(rightDataset) == 0: continue
             #在当前特征和该特征的切分下得到树桩模型
             tempModel = {}
             tempModel["spInd"] = featIndex
             tempModel["spValue"] = featValue
             #计算左右叶子节点的的值
             leftLeafValue = 0.0 ; rightLeafValue = 0.0
             for inst in leftDataset:
                 leftLeafValue = leftLeafValue + inst[-1]
             for inst in rightDataset:
                 rightLeafValue = rightLeafValue + inst[-1]
                 
             leftLeafValue = leftLeafValue / len(leftDataset)
             rightLeafValue = rightLeafValue / len(rightDataset)
             tempModel["left"] = leftLeafValue
             tempModel["right"] = rightLeafValue
             tempSquareError = caculateSquareError(tempModel, dataset)
             if tempSquareError < lowestSquareError:
                 lowestSquareError = tempSquareError; bestSplitFeatIndex = featIndex; bestFeatSplitValue = featValue; bestLeftLeafNode = leftLeafValue; bestRightLeafNode = rightLeafValue   
    retTree = {}
    retTree["spInd"] = bestSplitFeatIndex
    retTree["spValue"] = bestFeatSplitValue
    retTree["left"] = bestLeftLeafNode
    retTree["right"] = bestRightLeafNode
    return retTree

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
def binSplitDataset(dataset, featIndex, featValue):
    leftDataset = []
    rightDataset = []
    leftIndex = np.nonzero([inst[featIndex] < featValue for inst in dataset])[0]
    rightIndex = np.nonzero([inst[featIndex] >= featValue for inst in dataset] )[0]
    for i in leftIndex:
        leftDataset.append(dataset[i])
    for i in rightIndex:
        rightDataset.append(dataset[i])
    return leftDataset, rightDataset 

"""
计算样本集的平方差之和
@param stump
 回归树桩
@param dataset
 数据集
@return 平方误差和
"""
def caculateSquareError(stump, dataset):
    sumSquareError = 0.0
    for inst in dataset:
        yHat = regressionStumpPredict(stump, inst[0 : -1])
        sumSquareError = sumSquareError + (inst[-1] - yHat) * (inst[-1] - yHat)
    return sumSquareError
 
if __name__ == '__main__':
    dataset = loadSimpleDataset()
    datasetMat = np.mat(dataset)
    #plt.plot(datasetMat[:, 0], datasetMat[:, 1], "go-")
    #plt.show()
    
    model = train(dataset, 30)
    #print model
    #接下来算一下平方误差的和，与线性回归做比较
    sumSquareError = 0.0
    for inst in dataset:
        yHat = predict(model, inst[0 : -1])
        sumSquareError = sumSquareError + (inst[-1] - yHat) * (inst[-1] - yHat)
    print sumSquareError
    
    