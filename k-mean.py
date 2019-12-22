import matplotlib.pyplot as plt
from numpy import *
import random
import os
from sklearn.decomposition import PCA  



#读取文件
def loadDataSet(X_signals_paths):
    classLabelVector = ['x','y']
    x_signals = []
    # y_ = []
    returnMat = []
    for fname in  os.listdir(X_signals_paths):  # 读取文件下的txt文件
        # y_.append(txt2num(fname))
        signal_type_path = os.path.join(X_signals_paths, fname)
        # print(signal_type_path)
        with open(signal_type_path, 'r') as file:
            x_signals.append(
                # [row.strip().split(' ')[8:] for row in file]
                [array(serie, dtype=float32) for serie in [row.strip().split(' ') for row in file]]
            )
        #提取特征
        x_feature = Feature_extraction(array(x_signals))
        returnMat.append(x_feature)
    return array(returnMat), classLabelVector




#提取特征值
def  Feature_extraction(x_signals):
    #提取最大值
    x_max = x_signals.max()
    #提取最小值
    x_min = x_signals.min()  
    #提取平均值
    x_mean = x_signals.mean()
    #提取方差
    x_var = x_signals.var()
    #提取标准差
    x_std = x_signals.std()
    #提取中值
    x_median = median(x_signals)
    #求和
    x_sum = x_signals.sum()
    return array([x_max,x_min,x_mean,x_var,x_std,x_median,x_sum])



# 欧几里得距离
def edistance(v1, v2):
    result=0.0
    for i in range(len(v1)):
        result +=(v1[i]-v2[i])**2
    return sqrt(result)


# 特征值归一化
def autoNorm(dataSet):
    minVals = dataSet.min(0)            # 获取特征值最小值
    maxVals = dataSet.max(0)            # 获取特征值最大值
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #归一化
    return normDataSet, ranges, minVals


#k-mean聚类
def kcluster(rows, distance=edistance, k=3):
    normDataSet, ranges, minVals = autoNorm(rows)   # 归一化数据到0-1之间
    count = normDataSet.shape[0]                        # 数据总数
    randinfo = random.sample(range(0, count), k)
    clusters = [normDataSet[randinfo[i]] for i in range(len(randinfo))]  # 随机选取k个值作为聚类中心

    lastmatches = None
    for t in range(100):
        bestmatches = [[] for i in range(k)]

        # 寻找最近中心
        for j in range(count):
            row = normDataSet[j]
            bestmatch = 0
            for i in range(k):
                d = distance(row,clusters[i])
                if d < distance(row,clusters[bestmatch]): bestmatch = i
            bestmatches[bestmatch].append(j)

        # 如果没有变化则认为最佳，退出循环
        if bestmatches == lastmatches: break
        lastmatches = bestmatches

        # 移动聚类的中心
        for i in range(k):
            avgs = [0.0] * len(normDataSet[0])
            if len(bestmatches[i]) > 0:
                for rowid in bestmatches[i]:
                    for m in range(len(normDataSet[rowid])):
                        avgs[m] += normDataSet[rowid][m]
                for j in range(len(avgs)):
                    avgs[j] /= len(bestmatches[i])
                clusters[i] = avgs

    return bestmatches

def plot(dataMat, labelMat,bestmatches):
    xcord = [];ycord = []
    sumx1=0.0;sumy1=0.0;sumx2=0.0;sumy2=0.0;sumx3=0.0;sumy3=0.0
    midx = [];midy=[]
    for i in range(len(dataMat)):
        xcord.append(float(dataMat[i][0]));ycord.append(float(dataMat[i][1]))
    for i in range(len(bestmatches)):
        for j in bestmatches[i]:
            if(i==0):
                plt.scatter(xcord[j], ycord[j], color='red')
                sumx1 += xcord[j]
                sumy1 += ycord[j]
            if(i == 1):
                plt.scatter(xcord[j], ycord[j], color='green')
                sumx2 += xcord[j]
                sumy2 += ycord[j]
            if (i == 2):
                plt.scatter(xcord[j], ycord[j], color='black')
                sumx3 += xcord[j]
                sumy3 += ycord[j]
    midx.append(sumx1 / len(bestmatches[0]))
    midx.append(sumx2 / len(bestmatches[1]))
    midx.append(sumx3 / len(bestmatches[2]))
    midy.append(sumy1 / len(bestmatches[0]))
    midy.append(sumy2 / len(bestmatches[1]))
    midy.append(sumy3 / len(bestmatches[2]))
    plt.scatter(midx,midy, marker = '+',color='blue')
    plt.xlabel(labelMat[0]);plt.ylabel(labelMat[1])
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.show()

if __name__=='__main__':
    dataMat, labelMat = loadDataSet(r'C:/Users/lenovo】/Desktop/测试')        #读取文件
    bestmatches = kcluster(dataMat)
    pca = PCA(n_components=2)  
    newData = pca.fit_transform(dataMat)
    print('提取的特征：') 
    print(dataMat)
    print('降维后的数据：')
    print(newData)
    print('聚类的结果：')
    print(bestmatches)
    normDataSet, ranges, minVals = autoNorm(newData)
    plot(normDataSet, labelMat,bestmatches)
