# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 15:11:24 2016

@author: TuGuanghui

@E-mail: guanghuitu@gmail.com

实现K均值聚类算法
使用算法需要调节的参数： 1)K值
"""
import numpy as np

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
    return np.array(dataset)

"""
K-均值聚类算法
@param data
 数据集
@param K
 需要聚类的cluster的数量
@return 每个cluster的质心以及每个cluster对应的样本的下标
"""     
def kMeans(data, K):
    m, n = data.shape # m为样本数量， n为特征的数量
    clusters = {} #存放每个cluster对应的样本
    ##Setp 0: 随机选择K个点作为类别的质心
    indeies = np.random.randint(m, size = K)
    centroids = data[indeies,:]
    ##Setp 1: 聚类，直到质心不再变化
    while True:
        distances = np.zeros((m, K)) #初始化每个样本与各个类别的质心的相似度
        #计算所有样本与所有质心的相似度
        for i in range(m):
            for j in range(K):
                distances[i, j] = eulidDis(data[i, :], centroids[j])
        lables = np.argmin(distances, axis = 1) #得到每个样本的分配的label
        #计算新的质心
        count = 0 #计数变量，用来判断是否质心发生了改变
        for i in range(K):
            tempIndeies = np.nonzero(lables == i)[0] #得到第i个类别所包含样本的下标
            clusters[i] = tempIndeies
            tempCentroid = np.mean(data[clusters[i], :], axis = 0) #得到第i个类别的质心
            if np.sum(centroids[i] - tempCentroid) != 0:
                centroids[i] = tempCentroid
            else:
                count += 1
        if count == K:
            #如果count == K，说明质心没有改变，退出循环
            break

    return centroids, clusters

"""
实现K-means++聚类算法，克服原始的K-means对初值敏感的缺点
@param data
 数据集
@param K
 需要聚类的cluster的数量
@return 每个cluster的质心以及每个cluster对应的样本的下标
""" 
def kMeansPlusPlus(data, K):
    m, n = data.shape # m为样本数量， n为特征的数量
    clusters = {} #存放每个cluster对应的样本
    ##Setp 0: 选取初始质心，尽量使初始的聚类中心之间的相互距离要尽可能的远
    centroids = intialCentroid(data, K)
    ##Setp 1: 聚类，直到质心不再变化
    while True:
        distances = np.zeros((m, K)) #初始化每个样本与各个类别的质心的相似度
        #计算所有样本与所有质心的相似度
        for i in range(m):
            for j in range(K):
                distances[i, j] = eulidDis(data[i, :], centroids[j])
        lables = np.argmin(distances, axis = 1) #得到每个样本的分配的label
        #计算新的质心
        count = 0 #计数变量，用来判断是否质心发生了改变
        for i in range(K):
            tempIndeies = np.nonzero(lables == i)[0] #得到第i个类别所包含样本的下标
            clusters[i] = tempIndeies
            tempCentroid = np.mean(data[clusters[i], :], axis = 0) #得到第i个类别的质心
            if np.sum(centroids[i] - tempCentroid) != 0:
                centroids[i] = tempCentroid
            else:
                count += 1
        if count == K:
            #如果count == K，说明质心没有改变，退出循环
            break
    return centroids, clusters

"""
初始化聚类中心，尽量使初始的聚类中心之间的相互距离要尽可能的远
@param data
 数据集
@param K
 需要聚类的cluster的数量
@return 每个cluster的质心以及每个cluster对应的样本的下标
""" 
def intialCentroid(data, K):
    m, n = data.shape # m为样本数量， n为特征的数量
    ##每次选择一个质心，尽量使初始的聚类中心之间的相互距离要尽可能的远
    centroids = np.zeros((K, n))
    index = np.random.randint(m)
    centroids[0] = data[index, :] #随机选择第一个质心（中心）
    for i in range(1, K): #每次初始化一个质心
        distances = np.zeros((m, i))
        ##计算每个样本到聚类中心的距离
        for j in range(m):
            for l in range(i):
                distances[j, l] = eulidDis(data[j, :], centroids[l])
        nearestDistances = np.min(distances, axis = 1) #找到每个样本到最近中心点的距离
        sumDis = np.sum(nearestDistances) #加和
        ##选取一个质心，nearestDistances中距离大者，被选中为质心的概率大
        randomNum = np.random.rand(1)[0] * sumDis
        for p in range(m):
            randomNum = randomNum - nearestDistances[p]
            if randomNum <= 0:
                centroids[i] = data[p, :]
                break
    return centroids
           
"""
两个实例之间的欧几里德距离
@param inA
 实例A
@param inB
 实例B
@return 欧几里德距离
"""    
def eulidDis(inA, inB):
    eulidDis =  np.sqrt(np.sum(np.power(inA - inB, 2)))
    return eulidDis

"""
对数据集进行最小最大标准化,放缩到[0, 1]区间
@param dataset
 数据集
@param minVals
 每个特征对应的最小值
@param maxVals
 每个特征对应的最大值
@return 已标准化好的数据集
""" 
def minMaxStandarization(data, minVals, maxVals):
    staDataset = data - minVals 
    rangeVals = maxVals - minVals
    staDataset = staDataset / rangeVals
    return staDataset

"""
PS:原始的Kmeans存在三个缺点： 
1）对初值(起始质心的选择)敏感， 使用K-means++解决，初始的聚类中心之间的相互距离要尽可能的远.
2）K难确定， todo
3）只能发现圆形或者说球形的cluster, 使用密度聚类解决
"""

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    data = loadDataset(".\\datasets\\testSet2.txt", "\t")
    ##对数据进行标准化
    minVals = np.min(data, axis = 0)
    maxVals = np.max(data, axis = 0)
    data = minMaxStandarization(data, minVals, maxVals)
    plt.plot(data[:, 0], data[:, 1], "go")
    centroids, clusters = kMeans(data, 3)
    #centroids, clusters = kMeansPlusPlus(data, 3) #进行聚类
    plt.plot(centroids[:, 0], centroids[:, 1], "ro") #将质心显示出来
    plt.show()
    
    
    



