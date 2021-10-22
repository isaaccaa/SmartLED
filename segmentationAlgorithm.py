import librosa
import numpy as np
from math import floor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from hmmlearn import hmm
from random import shuffle
import scipy.stats



# ----------------------- CQT -----------------------
def cqt(y, fs, B, lowFreq, highFreq):
    numOfFilters    = int(np.log2(highFreq/lowFreq)*B)
    cqt = np.zeros((numOfFilters,1))
    blockSize = len(y)
    y = y.reshape(len(y),1)
    for i in range(1,numOfFilters):
        step    = 1/fs
        # w       = lowFreq * 2 ^ ((i-1)/B)
        w       = lowFreq * np.power(2,((i-1)/B))
        windowEnd  = floor(blockSize * lowFreq / w)-1
        # t       = list(range(0,windowEnd*step,step))
        # t       = np.asarray(t)
        t       = np.arange(0,(windowEnd+1)*step,step)
        filter  = np.exp(2 * np.pi * 1j* w * t)
        c = np.dot(filter, y[0:(len(filter))])
        c = c/(windowEnd+1)
        cqt[i]  = c
    return cqt.T
def amplitudeInDecibel(x):
    z = np.where(x==0)
    x[z] = 0.00001
    y = 20 * np.log10(abs(x)/0.00001)
    return y
# ----------------------- 求频谱包络 -----------------------
def SpectralEnvelope(y, fs, windowSize, hopSize, B, lowFreq, highFreq):
    """
    @description  :
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    # 总分块数
    N = len(y)
    n = floor((N - windowSize) / hopSize) + 1
    numOfFilters    = int(np.log2(highFreq/lowFreq)*B)
    # 计算包络
    x = np.zeros((numOfFilters, n),dtype=float)
    x = x.T
    for i in range(1,n):
        iStart = (i-1) * hopSize
        iEnd = iStart + windowSize -1
        cqti = cqt(y[iStart:iEnd], fs, B, lowFreq, highFreq)
        x[i] = cqti
    # 取对数
    x = amplitudeInDecibel(x)
    x = x.T
    return x
# ----------------------- L2规范化 -----------------------
def normalize(x):
    M = len(x)      # 矩阵行数
    N = len(x[0])   # 矩阵列数

    y = np.zeros((M, N))     # 初始化规范化后的y
    pow = np.zeros((1, N))   # 初始化能量

    for i in range(N):
        power = np.linalg.norm(x[:,i])
        if abs(power) > 0.000000001:
            y[:,i] = x[:,i] / power
            pow[0,i] = power
        else:
            pow[0,i] = 0

    power_max = np.amax(pow)
    if (power_max > 0.0000000001):
        pow = pow / power_max

    return y, pow
# ----------------------- PCA -----------------------
def pca20(x):
    pca = PCA(n_components=20)
    pca.fit(x)
    y = pca.fit_transform(x)
    return y
# ----------------------- 获取低级特征函数 -----------------------
def getLowLevelFeatures(y, sr, B, lowFreq, highFreq):
    """
    @description  :
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """

    # 计算整体节奏 BPM
    onset_env = librosa.onset.onset_strength(y, sr=sr)
    bpm = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
    bpm = bpm[0]
    # 计算 hopSize 和 windowSize
    hopSize = round(sr * 60 / bpm)
    windowSize = 3 * hopSize   
    # 使用 CQT 计算低级特征矩阵
    spectral_envelope = SpectralEnvelope(y, sr, windowSize, hopSize, B, lowFreq, highFreq)
    # 将低级特征矩阵归一化到[0,1]上
    spectrum, power = normalize(spectral_envelope)
    # 将特征降维至20维投影
    spectrumProjection = pca20(spectrum.T)
    # 组合出最终的特征矩阵
    featureMatrix = np.append(spectrumProjection.T,power,axis = 0)
    featureMatrix = featureMatrix.T
    return featureMatrix, bpm
# ----------------------- HMM状态 -----------------------
def getHMMStates(x, statenum):
    remodel = hmm.GaussianHMM(n_components=statenum, covariance_type="full")
    # x2 = x / np.amax(x)
    remodel.fit(x)
    z = remodel.predict(x)
    return z
# ----------------------- 直方图统计 -----------------------
def histogramStatistic(stateSequence, neighbouringBeatNum, stateNum):
    histogramNum    =  len(stateSequence)-neighbouringBeatNum+1
    histogramMatrix =  np.zeros((histogramNum, stateNum))
    for i in range(histogramNum):
        currentBlock = stateSequence[i:i+neighbouringBeatNum]
        histogramMatrix[i] = np.bincount(currentBlock, minlength=stateNum)

    return histogramMatrix

# ----------------------- 直方图聚类 -----------------------
def histogramClustering(histogramMatrix, kNum):
    kmeans = KMeans(n_clusters=kNum).fit(histogramMatrix)
    clusterLabels = kmeans.labels_
    theta = kmeans.cluster_centers_
    return theta, clusterLabels

def JS_divergence(p,q):
    # JS散度
    # M=(p+q)/2
    # return 0.5*scipy.stats.entropy(p,M)+0.5*scipy.stats.entropy(q, M)
    return np.linalg.norm(p-q)

def qFunctionMax(observations, y_estimation, K, index, theta, dML, Plambda):
    """
    @description  :
    计算Q函数最大值及对应的参数K
    ---------
    @param  :
    observations:   观测序列X
    y_estimation:   估计的y标签序列
    K           :   聚类簇的数目
    index       :   当前的遍历坐标
    theta       :   质心向量序列
    dML         :   参考文献选择16
    Plambda     :   固定惩罚，参考文献选0.02
    -------
    @Returns  :
    -------
    """

    qList = np.zeros(K)
    # 计算K簇数据的方差
    # 初始化空的数组存放K个簇的数据的方差
    variance = np.ones(K)
    # 通过np.where选取观测序列的K个自序列，计算其方差
    for i in range(K):
        # 获取估计序列中结果为第i个的索引，子序列为observations[index]
        index_K = np.where(y_estimation == i)
        sigma2 = np.var(observations[index_K])
        variance[i] = sigma2

    # 计算这第i个观测向量与K个质心的JS散度，为了美观性，把JS散度分开放而不是直接乘在上一步的方差上面
    JSdivergence = np.zeros(K)
    for i in range(K):
        JSdivergence[i] = JS_divergence(observations[index], theta[i])

    # 惩罚项
    punish = np.zeros(K)
    # 计算左右边界
    left = index - dML
    right = index + dML
    if (left < 0):
        left = 0
    if (right > len(y_estimation)):
        right = len(y_estimation)

    punish_index = list(range(left,right))
    # i != j
    punish_index.remove(index)
    # Sigma_{i \neq j} 和 c_{ij}部分
    punish_seq = y_estimation[punish_index]
    for i in range(K):
        # 按照克罗内克函数的意义，处理1-delta
        count_index = np.where(punish_seq != i)
        count = len(count_index[0])
        punish[i] = Plambda * count

    for i in range(K):
        qList[i] = np.exp(-JSdivergence[i]*variance[i]-punish[i])

    finalK = np.argmax(qList)
    qMax = (np.amax(qList))/(np.sum(qList))
    # print ('q:',qMax)
    return finalK, qMax

def singleEStep(observations, y_estimation, K, theta, dML, Plambda, iterE_max):
    pre_y = y_estimation

    order = list(range(len(y_estimation)))

    yChanged = True
    qFunction = np.zeros(len(y_estimation))
    iterTimes = 0
    # 终止迭代条件：y序列不再改变或到达最大迭代次数
    while (yChanged and iterTimes<iterE_max):
        iterTimes += 1
        # 以随机顺序遍历y估值序列
        shuffle(order)
        for i in range(len(order)):
            # def qFunctionMax(observations, y_estimation, K, index, theta, dML, Plambda):
            y_estimation[order[i]], qFunction[i] = qFunctionMax(observations=observations,y_estimation=y_estimation,K=K,index=order[i],theta=theta,dML=dML,Plambda=Plambda)
            # print ('q:',qFunction[i])
        if (pre_y == y_estimation).all() :
            yChanged = False
        else:
            yChanged = True
        pre_y = y_estimation
    return y_estimation , qFunction


def singleMStep(observations, y_estimation, qList, K,M):

    # 空的新聚类中心向量
    newtheta = np.zeros((K,M))
    for i in range(K):
        # 找出聚到第i个簇的子序列
        index = np.where(y_estimation == i)
        newtheta[i] = (np.matmul(qList[index],observations[index]))/(np.sum(qList[index]))
    return newtheta

def EMAlgorithm(observations, y_estimation, K, M, theta, dML, Plambda, iterE_max, iterEM_max):

    pre_theta = theta
    theta_Changed = True
    iterTimes = 0
    while (theta_Changed and iterTimes < iterEM_max):
        iterTimes += 1
        y_estimation, qList = singleEStep(observations, y_estimation, K, theta, dML, Plambda, iterE_max)
        theta = singleMStep(observations, y_estimation, qList, K, M)
        if (pre_theta == theta).all():
            theta_Changed = False
        else:
            theta_Changed = True
        pre_theta = theta

    return y_estimation, theta

# ----------------------- 获取分段点 -----------------------
def getBoundaries(labels,duration ,ly):
    lastValue = -1
    boundaries = []
    length = len(labels)
    for i in range(length):
        if labels[i] != lastValue:
            boundaries.append(i)
            lastValue = labels[i]
    boundaries.append(length)
    boundariesTime = np.asarray(boundaries,dtype=int)
    # boundariesTime = boundariesTime * duration / length
    boundariesTime = boundariesTime * ly // length
    # 返回分段点时间序列
    # return boundaries
    cueList = []
    for i in range(len(boundaries)-1):
        cueList.append([boundariesTime[i],boundariesTime[i+1],labels[boundaries[i]]])
    return cueList
