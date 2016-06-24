# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 09:58:40 2015

@author: TuGuanghui

@E-mail: guanghuitu@gmail.com

利用CART做回归树，采用平方误差和(Square Error)作为基准来划分数据集
注意遇到的二个问题：1、伪逆的问题 2、什么时候应该停止划分的问题，即预剪枝的问题
"""
import numpy as np
import matplotlib.pyplot as plt

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
@param datasetMat
 训练数据集
@return 训练好的CART回归树
""" 
def train(datasetMat):
    bestFeatIndex, bestFeatSplitValue = chooseBestFeaturetoSplit(datasetMat)
    if bestFeatIndex == None: 
        return bestFeatSplitValue
    retTree = {}
    retTree["spInd"] = bestFeatIndex
    retTree["spValue"] = bestFeatSplitValue
    leftDatasetMat, rightDatasetMat = binSplitDataset(datasetMat, bestFeatIndex,  bestFeatSplitValue)
    retTree["left"] = train(leftDatasetMat) #训练左子树
    retTree["right"] = train(rightDatasetMat) #训练右子树
    return retTree

"""
利用训练好的CART回归树预测
@param model
 训练好的CART回归树
@param inX
 需要预测的实例
@return 预测值
"""     
def predict(treeModel, inX):
    if isinstance(treeModel, dict):
        if inX[treeModel["spInd"]] <= treeModel["spValue"]:
            yHat = predict(treeModel["left"], inX) #左子树方向
        else:
            yHat = predict(treeModel["right"], inX) #右子树方向
    else:
        yHat = linearModelPredict(treeModel, inX)
    return yHat


"""
采用SE(平方误差和)评价标准来选择最佳的切分特征和该特征的最佳切分值
@param dataset
 训练集
@return 最好的切分特征和切分值
""" 
def chooseBestFeaturetoSplit(datasetMat):
    epsilon = 1 #阈值，当小于该阈值时，不进行划分数据集
    leastSamples = 3 #阈值，当划分后的左子树或者右子树中的样本集小于3时，不进行划分数据集
    m, n = np.shape(datasetMat)
    n = n -1 #让n等于特征的数量，故让n-1
    orginalSE = caculateSE(datasetMat)
    bestSE = np.inf
    bestFeatIndex = 0; bestFeatSplitValue = 0.0
    #对每一个特征的每一个取值的划分得到的MSE进行计算,找到bestFeatIndex，和bestFeatSplitValue
    for featIndex in range(n):
        for featValue in set(np.ravel(datasetMat[:,featIndex])): ##此步不科学（对数值型的特征），每个取值都要遍历一遍，可参照adaBoost代码修改
            leftDatasetMat, rightDatasetMat = binSplitDataset(datasetMat, featIndex, featValue)
            if (np.shape(leftDatasetMat)[0] < leastSamples) or (np.shape(rightDatasetMat)[0] < leastSamples): continue
            newSE = caculateSE(leftDatasetMat) + caculateSE(rightDatasetMat)
            if newSE < bestSE:
                bestSE = newSE; bestFeatIndex = featIndex; bestFeatSplitValue = featValue
    if(orginalSE - bestSE < epsilon):
        return None, buildLinearModel(datasetMat) #如果SE变化的很小，则不进行划分
    leftDatasetMat, rightDatasetMat = binSplitDataset(datasetMat, bestFeatIndex, bestFeatSplitValue)
    if (np.shape(leftDatasetMat)[0] < leastSamples) or (np.shape(rightDatasetMat)[0] < leastSamples):  #exit cond 3
        return None, buildLinearModel(datasetMat)
    return bestFeatIndex, bestFeatSplitValue

"""
计算平方误差和
@param dataset
 数据集
@return 平方误差和
"""
def caculateSE(datasetMat):
    weights = buildLinearModel(datasetMat) #weights[0]为b值
    SE = 0.0
    for insMat in datasetMat:
        insList = np.ravel(insMat).tolist()
        yHat = linearModelPredict(weights, insList[0:-1])
        error = np.power(yHat - insList[-1], 2)
        SE = SE + error
    return SE

"""
训练线性模型
@param datasetMat
 训练数据集
@return 训练好的线性模型
""" 
def buildLinearModel(datasetMat):
    m, n = np.shape(datasetMat)
    n = n- 1 #让n等于特征的数量，故让n-1
    YMat = datasetMat[:, -1]
    XMat = np.column_stack((np.ones((m,1)), datasetMat[:, 0:-1]))
    XMatTXMatI = (XMat.T*XMat + 0.0001 *np.eye(n+1) ).I #使用伪逆
    #weights = ((XMat.T*XMat).I)*(XMat.T)*YMat
    weights = XMatTXMatI * XMat.T*YMat
    return np.ravel(weights)
 
"""
利用训练好的线性模型预测
@param model
 训练好的线性模型
@param inX
 需要预测的实例
@return 预测值
"""
def linearModelPredict(weights, inX):
    yHat = 0.0
    for i in range(len(inX)):
        yHat = yHat + inX[i]*weights[i+1]
    yHat = yHat + weights[0]
    return yHat

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
def binSplitDataset(datasetMat, featIndex, featValue):
    leftDataset = []
    rightDataset = []
    leftIndex = np.nonzero(datasetMat[:,featIndex] <= featValue)[0]
    rightIndex = np.nonzero(datasetMat[:,featIndex] > featValue)[0]
    for i in leftIndex:
        leftDataset.append(np.ravel(datasetMat[i,:]).tolist())
    for i in rightIndex:
        rightDataset.append(np.ravel(datasetMat[i,:]).tolist())
    return np.mat(leftDataset), np.mat(rightDataset)


if __name__ == '__main__':
    dataset = loadDataset(".\\datasets\\exp2.txt")
    datasetMat = np.mat(dataset)
    plt.plot(datasetMat[:,0], datasetMat[:, 1], "bo")
    plt.show()
    myTree = train(datasetMat)
    print myTree
    for ins in datasetMat:
         yHat = predict(myTree, np.ravel(ins[0, 0:-1]))
         print "perdicted: " ,  yHat, " real: " , np.ravel(ins[0, -1])[0]
   
