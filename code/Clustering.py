from __future__ import print_function
from numpy import *
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
# 从文本中构建矩阵，加载文本文件，然后处理
def loadDataSet(fileName):  # 通用函数，用来解析以 tab 键分隔的 floats（浮点数）
    dataSet = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(',')
        fltLine =list(map(float, curLine))  # 映射所有的元素为 float（浮点数）类型
        dataSet.append(fltLine)
    return dataSet


# 计算两个向量的欧式距离（可根据场景选择）
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))  # la.norm(vecA-vecB)


# 为给定数据集构建一个包含 k 个随机质心的集合。随机质心必须要在整个数据集的边界之内，这可以通过找到数据集每一维的最小和最大值来完成。然后生成 0~1.0 之间的随机数并通过取值范围和最小值，以便确保随机点在数据的边界之内。
def randCent(dataMat, k):
    n = shape(dataMat)[1]  # 列的数量
    centroids = mat(zeros((k, n)))  # 创建k个质心矩阵
    for j in range(n):  # 创建随机簇质心，并且在每一维的边界内
        minJ = min(dataMat[:, j])  # 最小值
        rangeJ = float(max(dataMat[:, j]) - minJ)  # 范围 = 最大值 - 最小值
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))  # 随机生成
    return centroids


# k-means 聚类算法
# 该算法会创建k个质心，然后将每个点分配到最近的质心，再重新计算质心。
# 这个过程重复数次，知道数据点的簇分配结果不再改变位置。
# 运行结果（多次运行结果可能会不一样，可以试试，原因为随机质心的影响，但总的结果是对的， 因为数据足够相似，也可能会陷入局部最小值）
def kMeans(dataMat, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataMat)[0]  # 行数
    clusterAssment = mat(zeros((m, 2)))  # 创建一个与 dataMat 行数一样，但是有两列的矩阵，用来保存簇分配结果
    centroids = createCent(dataMat, k)  # 创建质心，随机k个质心
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # 循环每一个数据点并分配到最近的质心中去
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :],
                                  dataMat[i, :])  # 计算数据点到质心的距离
                if distJI < minDist:  # 如果距离比 minDist（最小距离）还小，更新 minDist（最小距离）和最小质心的 index（索引）
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:  # 簇分配结果改变
                clusterChanged = True  # 簇改变
                clusterAssment[i, :] = minIndex, minDist**2  # 更新簇分配结果为最小质心的 index（索引），minDist（最小距离）的平方
        # print(centroids)
        for cent in range(k):  # 更新质心
            ptsInClust = dataMat[nonzero(
                clusterAssment[:, 0].A == cent)[0]]  # 获取该簇中的所有点
            centroids[cent, :] = mean(
                ptsInClust, axis=0)  # 将质心修改为簇中所有点的平均值，mean 就是求平均值的
    return centroids, clusterAssment




# def testBasicFunc():
#     # 加载测试数据集
#     dataMat = mat(loadDataSet('data/10.KMeans/testSet.txt'))
#
#     # 测试 randCent() 函数是否正常运行。
#     # 首先，先看一下矩阵中的最大值与最小值
#     print('min(dataMat[:, 0])=', min(dataMat[:, 0]))
#     print('min(dataMat[:, 1])=', min(dataMat[:, 1]))
#     print('max(dataMat[:, 1])=', max(dataMat[:, 1]))
#     print('max(dataMat[:, 0])=', max(dataMat[:, 0]))
#
#     # 然后看看 randCent() 函数能否生成 min 到 max 之间的值
#     print('randCent(dataMat, 2)=', randCent(dataMat, 2))
#
#     # 最后测试一下距离计算方法
#     print(' distEclud(dataMat[0], dataMat[1])=', distEclud(dataMat[0], dataMat[1]))


def testKMeans():
    # 加载测试数据集
    # with open('E:///data//abide1//atten.txt') as filein, open(
    #         'E:/data/abide1/attenreally.txt', 'w') as fileout:
    #     for line in filein:
    #         line = line.replace("[", "")
    #         line = line.replace("]", "")
    #         fileout.write(line)
    dataMat = mat(loadDataSet('D://codee///data//abide1//attenmew.csv'))
    pca = PCA(n_components=2)  # 提取两个主成分，作为坐标轴
    pca.fit(dataMat)
    data_pca = pca.transform(dataMat)
    data_pca = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])
    print('pac:', data_pca)
    scaler =StandardScaler()
    # normal_data_pca = scaler.fit_transform(data_pca)
    # print(type(normal_data_pca))
    # 将数据进行0-1标准化
    # normalized_data = scaler.fit_transform(dataMat)
    # print('df:', normalized_data)

    # 该算法会创建k个质心，然后将每个点分配到最近的质心，再重新计算质心。
    # 这个过程重复数次，知道数据点的簇分配结果不再改变位置。
    # 运行结果（多次运行结果可能会不一样，可以试试，原因为随机质心的影响，但总的结果是对的， 因为数据足够相似）
    #11111111111第一种方法
    myCentroids, clustAssing = kMeans(data_pca.values, 3)
    data_pca.insert(data_pca.shape[1], 'labels', np.array(clustAssing)[:, 0])
    print('my:', myCentroids)

    ##2222222222
    # K-means聚类
    kms = KMeans(n_clusters=3)
    data_fig = kms.fit(data_pca.values)  # 模型拟合
    centers = kms.cluster_centers_  # 计算聚类中心
    labs = kms.labels_  # 为数据打标签
    print('lab:', labs)


    # pca = PCA(n_components=2)
    # pca.fit(myCentroids)
    # data_pca_centers = pca.transform(myCentroids)
    # data_pca_centers = pd.DataFrame(data_pca_centers, columns=['PC1', 'PC2'])
    # normal_myCentroids = scaler.fit_transform(myCentroids)
    # print(normal_myCentroids)
    # print('cf',np.array(clustAssing)[:, 0])
    # print('c:', clustAssing)
    # print('centroids=', myCentroids)
    # print(dataMat)
    # print('345:', np.array(dataMat)[:, 1])
    # plt.scatter(np.array(normalized_data)[:, 1], np.array(normalized_data)[:, 0], c=np.array(clustAssing)[:, 0])
    # plt.scatter(np.array(normal_myCentroids)[:, 1],np.array(normal_myCentroids)[:, 0], c="r")
    # plt.show()
    # print('centroids=', myCentroids)



    # ------------------------下面介绍如何绘制聚类散点图-----------------------------
    # 对二分类的散点图绘制，网上教程很多，此篇文章主要介绍多分类的散点图绘制问题
    # 首先，对原数据进行 PCA 降维处理，获得散点图的横纵坐标轴数据
    # pca = PCA(n_components=2)  # 提取两个主成分，作为坐标轴
    # pca.fit(dataMat)
    # data_pca = pca.transform(dataMat)
    # data_pca = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])
    # print('data:', data_pca)
    # # normal_data_pca = scaler.fit_transform(data_pca)
    # # normal_data_pca = pd.DataFrame(normal_data_pca)
    # data_pca.insert(data_pca.shape[1], 'labels', np.array(clustAssing)[:, 0])
    # # print('no:',normal_data_pca)
    #
    #
    # # centers pca 对 K-means 的聚类中心降维，对应到散点图的二维坐标系中
    # pca = PCA(n_components=2)
    # pca.fit(myCentroids)
    # data_pca_centers = pca.transform(myCentroids)
    # data_pca_centers = pd.DataFrame(data_pca_centers, columns=['PC1', 'PC2'])
    # normal_data_pca_centers = scaler.fit_transform(data_pca_centers)
    # print(normal_data_pca_centers)
    # Visualize it:
    print('frg:', np.array(data_pca)[:, 2])
    plt.figure(figsize=(8, 6))
    plt.scatter(np.array(data_pca.values)[:, 0], np.array(data_pca.values)[:, 1],  c= labs)
    plt.scatter( np.array(centers)[:, 0], np.array(centers)[:, 1], marker='o', s=55, c='r')
    plt.show()
    #轮廓系数
    score= silhouette_score(data_pca,np.array(data_pca)[:, 2])
    print('s:',  score)

    # 手肘图法1——基于平均离差
    K = range(1, 18)
    meanDispersions = []
    for k in K:
        kemans = KMeans(n_clusters=k, init='k-means++')
        kemans.fit(dataMat)
        # 计算平均离差
        m_Disp = sum(np.min(cdist(dataMat, kemans.cluster_centers_, 'euclidean'), axis=1)) / \
                 dataMat.shape[0]
        meanDispersions.append(m_Disp)

    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使折线图显示中文

    plt.plot(K, meanDispersions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('平均离差')
    plt.title('')
    plt.show()

    # 手肘图法2——基于SSE
    distortions = []  # 用来存放设置不同簇数时的SSE值
    for i in range(1, 15):
        kmModel = KMeans(n_clusters=i)
        kmModel.fit(dataMat)
        distortions.append(kmModel.inertia_)  # 获取K-means算法的SSE
    # 绘制曲线
    plt.plot(range(1, 15), distortions, marker="o")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel("簇数量")
    plt.ylabel("簇内误差平方和(SSE)")
    plt.show()


if __name__ == "__main__":

    # 测试基础的函数
    # testBasicFunc()0

    # 测试 kMeans 函数
    testKMeans()