from random import shuffle
import joblib
from sklearn import svm
import numpy as np
import librosa
from model.MSVR import MSVR
from model.utility import create_dataset, rmse

# ----------------------- 获取各分段的片段 -----------------------
def getSegmentPart(boundaries,kNum,percent):
    # 按各个标签归类
    segment = []
    durationTime = []
    for i in range(kNum):
        temp = []
        alltime = 0
        for item in boundaries:
            if item[2] == i:
                temp.append(item)
                alltime += item[1] - item[0]
        segment.append(temp)
        durationTime.append(alltime*percent)

    # 选取片段
    for i in range(kNum):
        shuffle(segment[i])
        select = -1
        selectTime = 0
        while (selectTime < durationTime[i]):
            select += 1
            selectTime += (segment[i][select][1] - segment[i][select][0])
        select = 1 if select==0 else select
        segment[i] = segment[i][0:select]
    return segment
        
# ----------------------- 获取音乐分段序列 -----------------------
def getSegmentSequence(segment, kNum, y):
    Sequence = []
    for i in range(kNum):
        tempSeq = np.zeros(1)
        # 将分段信息中的序列对应到y中，加入到tempSeq
        for item in segment[i]:
            tempSeq = np.append(tempSeq,y[item[0]:item[1]])
        tempSeq = np.delete(tempSeq,0)
        Sequence.append(tempSeq)
    # 返回K个音乐序列
    return Sequence

# ----------------------- 提取一段音乐序列的特征 -----------------------
def getEmotionFeature(y, sr):
    S = np.abs(librosa.stft(y))

    # Extracting Features
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)          # 节奏
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]     # 频谱质心
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)    # 谱对比度
    mfcc = librosa.feature.mfcc(y=y, sr=sr)                     # MFCC
    chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)          # 半音CQT谱
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)       # 半音谱质心
    zcr = librosa.feature.zero_crossing_rate(y)                 # 短时平均过零率
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)      # 频谱衰减
    
    # Transforming Features
    features = []
    features.append(tempo)  # tempo

    features.append(np.mean(cent))  # cent
    features.append(np.std(cent))
    features.append(np.var(cent))

    features.append(np.mean(contrast))  # contrast
    features.append(np.std(contrast))
    features.append(np.var(contrast))

    features.append(np.mean(mfcc))  # mfcc
    features.append(np.std(mfcc))
    features.append(np.var(mfcc))
    
    features.append(np.mean(chroma_cq))  # chroma_cq
    features.append(np.std(chroma_cq))
    features.append(np.var(chroma_cq))

    features.append(np.mean(chroma_cens))  # chroma_cens
    features.append(np.std(chroma_cens))
    features.append(np.var(chroma_cens))

    features.append(np.mean(zcr))  # zcr
    features.append(np.std(zcr))
    features.append(np.var(zcr))

    features.append(np.mean(rolloff))  # rolloff
    features.append(np.std(rolloff))
    features.append(np.var(rolloff)) 
       
    features = np.asarray(features)

    features[np.isnan(features)] = 0    # 清理NaN

    return features

# ----------------------- 提取各分段的特征 -----------------------
def getSegmentFeature(musicSequence, kNum, sr):
    characteristicsNumber = 22
    feature = np.zeros((kNum,characteristicsNumber))
    for i in range(kNum):
        feature[i] = getEmotionFeature(musicSequence[i], sr)
    return feature

# ----------------------- 获取各分段的情感判断 -----------------------
def getSegmentEmotion(seqFeature):
    clf = joblib.load("music_audio.model")
    seqEmotions = clf.predict(seqFeature)
    # print(clf.predict(seqFeature))
    # for item in seqFeature: 
    #     print(clf.predict(item))
    return seqEmotions

# ----------------------- 主情感 -----------------------
def getMainEmotion(seqEmotions):
    mainEmotions = []
    for item in seqEmotions:
        emotion = np.argsort(item)
        maxEmotion = emotion[len(emotion)-1]
        secondaryEmotion = emotion[len(emotion)-2]
        mainEmotions.append([maxEmotion,secondaryEmotion])
    return mainEmotions
