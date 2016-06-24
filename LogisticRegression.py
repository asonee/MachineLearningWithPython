# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 08:29:31 2015

@author: TuGuanghui

@E-mail: guanghuitu@gmail.com

实现(带正则化项的)Logistic回归算法
使用算法需要调节的超参数: 1)学习率 2）权重衰减系数 3）最大迭代次数 4）使用mini批梯度下降，计算cost和导数选用的样本数
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
    return np.array(dataset)

"""
通过批梯度下降算法训练Logistic
@param dataset
 训练集
@param theta
 需训练得出的参数
@param alpha
 学习率
@param lambda_
 权重衰减系数,用来权衡bias和variance
@param maxIter
 最大迭代次数
@return 最优的theta
"""
def trainByBatchGradientDescent(dataset, theta, alpha = 0.01, lambda_ = 0, maxIter = 100):
    data = dataset[:, 0 : -1]
    labels = dataset[:, -1]
    costes = np.zeros(maxIter)
    for i in range(maxIter):
        #每一轮迭代利用所有的训练集计算代价和导数
        cost , grad = logisticCost(theta, lambda_, data, labels)
        theta = theta - alpha * grad #更新梯度
        costes[i] = cost
    #将每次计算得出的损失函数的值画图出来，看看是否已经收敛
    plt.plot(np.arange(maxIter), costes, "r-")
    plt.show()
    return theta

"""
通过随机梯度下降算法训练Logistic
@param dataset
 训练集
@param theta
 需训练得出的参数
@param alpha
 学习率
@param lambda_
 权重衰减系数,用来权衡bias和variance
@param maxIter
 最大迭代次数
@return 最优的theta
"""
def trainByStochasticGradientDescent(dataset, theta, alpha = 0.01, lambda_ = 0, maxIter = 100):
    data = dataset[:, 0 : -1]
    labels = dataset[:, -1]
    costes = np.zeros(maxIter)
    m = data.shape[0]
    for i in range(maxIter):
        #每次迭代，仅随机选取一个样本用来计算cost和导数
        index = np.random.randint(m)
        cost , grad = logisticCost(theta, lambda_, data[index, :].reshape((1, -1)), labels[index].reshape((1)))
        theta = theta - alpha * grad #更新梯度
        costes[i] = cost
    #将每次计算得出的损失函数的值画图出来，看看是否已经收敛
    plt.plot(np.arange(maxIter), costes, "r-")
    plt.show()
    return theta

"""
通过mini批梯度下降算法训练Logistic，为批梯度下降和随机梯度下降算法的折中
@param dataset
 训练集
@param theta
 需训练得出的参数
@param alpha
 学习率
@param lambda_
 权重衰减系数,用来权衡bias和variance
@param maxIter
 最大迭代次数
@param miniBatchNum
 每次选用的样本的数量，当miniBatchNum = 1时，即为随机梯度下降
@return 最优的theta
"""
def trainByMiniBatchGradientDescent(dataset, theta, alpha = 0.01, lambda_ = 0, maxIter = 100, miniBatchNum = 10):
    data = dataset[:, 0 : -1]
    labels = dataset[:, -1]
    costes = np.zeros(maxIter)
    m = data.shape[0]
    for i in range(maxIter):
        #随机选取一个样本用来计算cost和导数
        indeies = np.random.randint(m, size = miniBatchNum)
        cost , grad = logisticCost(theta, lambda_, data[indeies, :].reshape((miniBatchNum, -1)), labels[indeies].reshape((miniBatchNum)))
        theta = theta - alpha * grad #更新梯度
        costes[i] = cost
    #将每次计算得出的损失函数的值画图出来，看看是否已经收敛
    plt.plot(np.arange(maxIter), costes, "r-")
    plt.show()
    return theta

"""
利用训练好的Logistic分类
@param theta
 训练好的参数
@param inX
 需要分类的实例
@return 类别
"""     
def classify(theta, inX):
    W = theta[1 : ]
    b = theta[0]
    z = W.dot(inX.T) + b
    yHat = sigmoid(z)
    predLabels = []
    for temp in yHat:
        if temp > 0.5:
            predLabels.append(1)
        else:
            predLabels.append(0)
    return np.array(predLabels)
   
"""
计算Logistic代价函数的梯度和损失
@param theta
 参数向量
@param lambda_
 权重衰减系数
@param data
 训练数据
@param labels
 训练数据标签
@return 损失和梯度
"""
def logisticCost(theta, lambda_, data, labels):
    m = data.shape[0] #样本总数
    #从theta中提取出W和b
    W = theta[1 : ] #权重 or weights
    b = theta[0] #偏值 or bias
    data = data.T #进行转置，使得每一列代表一个样本
    #计算预测值
    z = W.dot(data) + b
    yHat = sigmoid(z)
    error = labels - yHat #计算真实值和预测值之间的误差
    #计算损失函数, + 0.0001是为了防止溢出，若要进行梯度检验，需将0.0001删掉
    cost = -(1.0 / m) * (labels.dot(np.log(yHat + 0.0001)) + (1.0 - labels).dot(np.log(1.0 - yHat + 0.0001))) + lambda_ * W.dot(W) / 2.0 
    #计算导数
    WGrad = -(1.0 / m) * data.dot(error) + lambda_ * W
    bGrad = -(1.0 / m) * np.sum(error)
    bGrad = np.array([bGrad]) #转换成np.array
    #将WGrad和bGrad放入一个向量中
    grad = np.concatenate((bGrad, WGrad))
    return cost, grad

"""
梯度检验函数，用来确定logisticCost中的求导是否正确实现
"""
def gradientCheck():
    import gradient
    dataset = np.random.randn(12).reshape(3, 4)
    labels = np.array([1, 1, 0])
    n = dataset.shape[1]
    theta = np.random.randn(n+1) #初始化参数
    lambda_ = 0.01
    cost, grad = logisticCost(theta, lambda_, dataset, labels)
    J = lambda x : logisticCost(x, lambda_, dataset, labels)
    numGrad = gradient.computeNumericGradient(J, theta)
    gradient.checkGradient(grad, numGrad)
      
#计算sigmoid函数值    
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-1.0 * x))

if __name__ == '__main__':
    dataset = loadDataset(".\\datasets\\horseColicTraining.txt")
    m = dataset.shape[0]
    n = dataset.shape[1] - 1
    ##Step 0: 初始化参数
    theta = np.ones(n + 1) #初始化参数
    lambda_ = 5 #权重衰减系数
    alpha = 0.02 #学习率
    maxIter = 200 #最大迭代次数
    ##Step 1: 训练Logistic
    #optTheta = trainByBatchGradientDescent(dataset, theta, alpha, lambda_, maxIter)
    #optTheta = trainByStochasticGradientDescent(dataset, theta, alpha, lambda_, 100)
    optTheta = trainByMiniBatchGradientDescent(dataset, theta, alpha, lambda_, maxIter, miniBatchNum = int(m * 0.9))
    preLables = classify(optTheta, dataset[:, 0 : -1])
    print "traing Accuray: ", np.sum(preLables == dataset[:, -1]) / float(dataset.shape[0])
    ##Step 2: 测试Logistic
    dataset4Test = loadDataset(".\\datasets\\horseColicTest.txt")
    preLables = classify(optTheta, dataset4Test[:, 0 : -1])
    print "testing Accuray: ", np.sum(preLables == dataset4Test[:, -1]) / float(dataset4Test.shape[0])


