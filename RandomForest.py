# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 22:26:01 2016

@author: TuGuanghui

@E-mail: guanghuitu@gmail.com

实现随机森林算法， 森林里面的基分类器采用CART分类树
随机体现在：1）样本集的随机选择（行抽样） 2）随机选择特征子集（列抽样在每棵树每个节点的划分上随机选择k个特征作为候选特征）注意：2）的代码实现在chooseBestFeaturetoSplit()里面
使用算法需要调节的超参数：1）树的数目 2）训练决策树的每个结点时，特征子集中特征的个数，即候选特征的数目
"""

import numpy as np
import operator
import random

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
训练随机森林算法
@param dataset
 训练集
@param M
 需要训练的弱分类器的数目
@param candidateFeatNum
 训练决策树的每个结点,即划分数据集时，候选特征的数目
@param isCalculateEstimate
 是否进行OOB错误估计
@return 训练好的模型
""" 
def buildClassifier(dataset, M = 100, candidateFeatNum = 1, isCalculateEstimate = True):
    if isCalculateEstimate:
        return buildAndOOBEstimate(dataset, M, candidateFeatNum)
    else:
        return build(dataset, M, candidateFeatNum)
        

"""
训练随机森林算法，不计算out of bag error estimate
@param dataset
 训练集
@param M
 需要训练的弱分类器的数目
@param candidateFeatNum
 训练决策树的每个结点,即划分数据集时，候选特征的数目
@return 训练好的模型
""" 
def build(dataset, M, candidateFeatNum):
    ensembleModel = {}
    for i in range(M):
        print "正在训练：", i
        newDataset = bootstrap(dataset)
        currentBaseModel = buildBaseClassifier(newDataset, candidateFeatNum)
        if isinstance(currentBaseModel, dict): 
            #如果训练出来的基模型不是一个叶子节点,即当数据量小的时候，会出现bootstrap出来的样本全为一个类别，此种得到的树应丢弃
            ensembleModel[i] = currentBaseModel
    return ensembleModel

"""
训练随机森林算法，并计算out of bag error estimate（包外估计）
@param dataset
 训练集
@param M
 需要训练的弱分类器的数目
@param candidateFeatNum
 训练决策树的每个结点,即划分数据集时，候选特征的数目
@return 训练好的模型
""" 
def buildAndOOBEstimate(dataset, M, candidateFeatNum):
    m = len(dataset)
    classLabels = list(set([inst[-1] for inst in dataset]))  #声明该变量，是为了得到error matrix
    K = len(classLabels)
    ensembleModel = {}
    errorMatrix = np.zeros((m, K)) #该矩阵用来存储OOB样本i对应的基分类器对它的分类结果，具体分到K个类别的那个类
    for i in range(M):
        print "正在训练：", i
        bootstraped, OOB = boostrap4OOBE(m)
        newDataset = []
        for i in bootstraped:
            newDataset.append(dataset[i])
        currentBaseModel = buildBaseClassifier(newDataset, candidateFeatNum)
        if isinstance(currentBaseModel, dict): 
            #如果训练出来的基模型不是一个叶子节点,即当数据量小的时候，会出现bootstrap出来的样本全为一个类别，此种得到的树应丢弃
            ensembleModel[i] = currentBaseModel
            #接下来对该基分类器的OOB利用该基分类器分类，并将结果保存进errorMatrix
            for i in OOB:
                #print "OOB: ", OOB
                inst = dataset[i]
                preClassLabel = baseClassifierClassify(currentBaseModel, inst[0 : -1])
                for j in range(K):
                    if preClassLabel == classLabels[j]:
                        errorMatrix[i][j] += 1
    #下面利用errorMatrix计算oob error
    right = 0
    wrong = 0
    flag = True
    for i in range(m):
        flag = False
        realClassLabel = dataset[i][-1]
        realClassLabelIndex = classLabels.index(realClassLabel)
        for j in range(K):
            if errorMatrix[i][realClassLabelIndex] < errorMatrix[i][j]:
                flag = True
                wrong += 1
        if((not flag) and errorMatrix[i][realClassLabelIndex] != 0):
            #errorMatrix[i][realClassLabelIndex] != 0,
            #对于每一个是OOB的样本，这样写是怕有的样本从来都没有当过OOB样本errorMatrix[i, j] == 0
            right += 1
    print "oob error estimate is: ", wrong / float(right + wrong)
    #print "oob error estimate2 is: ", wrong / float(m)
    print "total samples： ",m
    print "OOB samples: ", (right + wrong)
    return ensembleModel
    
"""
利用训练好的随机森林分类
@param model
 训练好的随机森林模型
@param inX
 需要分类的实例
@return 类别
"""     
def classify(model, inX):
    result = {}
    for key in model.keys():
        predictR = baseClassifierClassify(model[key], inX)
        result[predictR] = result.get(predictR, 0) +1
        #if predictR not in result.keys():
            #result[predictR] = 0
        #result[predictR] = result[predictR] +1
    sortedresult = sorted(result.iteritems(), key = operator.itemgetter(1), reverse = True)
    return sortedresult[0][0]

"""
使用自助法产生新的样本
@param dataset
 样本集
@return 新样本集
""" 
def bootstrap(dataset):
    retDataset = [];
    for i in range(len(dataset)):
        retDataset.append(dataset[random.randint(0, (len(dataset)-1))])
    return retDataset;

"""
使用自助法产生样本for OOB error estimate 
@param m
 样本总数
@return 被抽样到的和未被抽样到的样本的下标
""" 
def boostrap4OOBE(m):
    OOB = []
    bootstraped = []
    for i in range(m):
        bootstraped.append(random.randint(0, (m - 1)))
    for i in range(m):
        if bootstraped.count(i) == 0:
            OOB.append(i)
    return bootstraped, OOB

"""
训练基分类器即完全长成的CART分类树
@param dataset
 训练集
@param candidateFeatNum
 训练决策树的每个结点,即划分数据集时，候选特征的数目
@return 训练好的基分类器
""" 
def buildBaseClassifier(dataset, candidateFeatNum):
    classLabels = [inst[-1] for inst in dataset]
    if classLabels.count(classLabels[0]) == len(classLabels):
        #如果该数据集中的所有样本的标签一样，则不进行划分
        return classLabels[0]
    #if np.shape(dataset)[0] < 2:
        #return majorityVot(dataset)
    bestFeatIndex, bestFeatSplitValue = chooseBestFeaturetoSplit(dataset, candidateFeatNum)
    if bestFeatIndex == None: 
        return bestFeatSplitValue
    retTree = {}
    retTree["spInd"] = bestFeatIndex
    retTree["spValue"] = bestFeatSplitValue
    leftDatasetMat, rightDatasetMat = binSplitDataset(dataset, bestFeatIndex,  bestFeatSplitValue)
    
    retTree["left"] = buildBaseClassifier(leftDatasetMat, candidateFeatNum)
    retTree["right"] = buildBaseClassifier(rightDatasetMat, candidateFeatNum)
    return retTree
    
"""
利用训练好的基分类器进行分类
@param model
 训练好的基分类器
@param inX
 需要分类的实例
@return 类别
"""     
def baseClassifierClassify(treeModel, inX):
    featIndex =  treeModel["spInd"]
    featSplitValue = treeModel["spValue"]
    if inX[featIndex] <= featSplitValue:
        if(isinstance(treeModel["left"], dict)):
            return baseClassifierClassify(treeModel["left"], inX)
        else:
            return treeModel["left"]
    else:
        if(isinstance(treeModel["right"], dict)):
            return baseClassifierClassify(treeModel["right"], inX)
        else:
            return treeModel["right"]

"""
采用GINI系数评价标准来选择最佳的切分特征和该特征的最佳切分值
@param dataset
 训练集
@param candidateFeatNum
 训练决策树的每个结点,即划分数据集时，候选特征的数目
@return 最好的切分特征和切分值
""" 
def chooseBestFeaturetoSplit(dataset, candidateFeatNum):
    epsilon = 0 #阈值，当小于该阈值时，不进行划分数据集
    m, n = np.shape(dataset)
    n = n -1 #让n等于特征的数量，故让n-1
    orginalGINI = caculateGINI(dataset)
    bestGINI = np.inf
    bestFeatIndex = 0; bestFeatSplitValue = 0.0
    #对每一个特征的每一个取值的划分得到的MSE进行计算,找到bestFeatIndex，和bestFeatSplitValue
    #对列进行抽样
    #N = int(math.log(n, 2))  #候选特征集中特征数目,推荐数目为log2(n)
    N = candidateFeatNum
    randomFeatInds = []
    for i in range(N):
        randomFeatInds.append(random.randint(0, (n-1))) #需要修改不让其抽到相同的特征
    for featIndex in randomFeatInds:
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
        #epsilon == 0此处是为了防止噪声数据，导致gini系数没有变，例如当前dataset为[1,1,0,-1],[1,1,0,1]，不管怎样都不能把它们分开
        #若不这样的话，会导致leftDataset为空,这个并不是剪枝
        #还有的话，就是怕选到的特征不能将数据集划分
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

if __name__ == '__main__':
    #dataset = loadSimpleDataset()
    #dataset = loadDataset(".\\datasets\\horseColicTraining2.txt") #最佳参数 树的数目100， 随机选择的特征数目 1个
    dataset = loadDataset(".\\datasets\\horseColicAll2.txt") #最佳参数 树的数目100， 随机选择的特征数目 1个
    
    rf =  buildClassifier(dataset, M = 100, candidateFeatNum = 1, isCalculateEstimate = True)
    """
    #datasetForTest = dataset
    datasetForTest =  loadDataset(".\\datasets\\horseColicTest2.txt")   
    correct = 0; wrong = 0
    for inst in dataset:
        if classify(rf, inst[0:-1]) == inst[-1]:
            correct += 1
        else:
            wrong += 1
    print "traning error rate is\t", float(wrong)/(wrong+correct)
    correct = 0; wrong = 0
    for inst in datasetForTest:
        if classify(rf, inst[0:-1]) == inst[-1]:
            correct += 1
        else:
            wrong += 1
    print "testing error rate is\t", float(wrong)/(wrong+correct)
    """
    