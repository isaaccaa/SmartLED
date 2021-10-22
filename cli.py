# This is a cli version for this project. It will NOT lanuch the GUI. 
# It seems more suitable for debug.

from segmentationAlgorithm import *
from sentimentAnalysis import *
from createCueList import *

import pprint
from matplotlib import pyplot as plt 

if __name__ == '__main__':
    # parameters
    B                   = 24
    lowFreq             = 62.5
    highFreq            = 16000
    stateNum            = 10
    neighbouringBeatNum = 21
    kNum                = 3
    percent             = 1

    # ====Draw====
    fig, ax = plt.subplots()
    plt.rcParams["font.family"] = 'Times New Roman'
    fig.set_size_inches(20,8)


    # Path of Music file. If you are using UNIX system, maybe should replace \ by /
    filename = 'TestMusic\XYZ.mp3'

    y, sr = librosa.load(path = filename)

    # ====Draw====
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
    wf = plt.subplot(411)
    wf.plot(y)
    plt.title('Music Waveform Figure',fontsize=22)
    wf.set_xticks([])
    # XYZ
    ref0 =np.zeros(11)
    ref1 = np.ones(21)
    ref2 = np.ones(108)
    ref2 = ref2+1
    ref3 = np.ones(20)
    ref4 = np.ones(64)
    ref4 = ref4 + 1
    ref5 = np.ones(22)
    ref6 = np.zeros(18)
    ref = np.concatenate((ref0,ref1,ref2,ref3,ref4,ref5,ref6))
    ref = np.concatenate((ref0,ref1,ref2,ref3,ref4,ref5,ref6))
    rf = plt.subplot(412)
    rf.plot(ref)
    plt.title('Reference',fontsize=22)
    rf.set_xticks([])


    fs = sr                 # I used a lot of 'fs' instand of sr in the following code , and I'm tired to replace them.
    # Length of the Music
    duration = librosa.get_duration(filename=filename)  
    # Get the 21-dimensional low-level feature matrix of the music
    featureMatrix, bpm = getLowLevelFeatures(y, fs, B, lowFreq, highFreq)
    # Estimating HMM model and Predict HMM state sequence
    seq = getHMMStates(featureMatrix, stateNum)
    # histogramStatistic
    histogramMatrix = histogramStatistic(seq,neighbouringBeatNum,stateNum)
    # K-Means, the base of following part, also as a baseline algorithm
    theta, idx = histogramClustering(histogramMatrix, kNum)

    # ====Draw====
    km = plt.subplot(413)
    km.plot(idx)
    plt.title('K-Means Clustering',fontsize=22)
    km.set_xticks([])


    # Make a better segmention
    idx2, theta2 = EMAlgorithm(histogramMatrix,idx,kNum,stateNum,theta,neighbouringBeatNum,0.8,100,100)

    # ====Draw====
    cc = plt.subplot(414)
    cc.plot(idx)
    plt.title('Constraint Clustering',fontsize=22)
    cc.set_xticks([])

    plt.show()
    fig.savefig('img/XYZ.eps',dpi=600,format='eps')

    # get the boundaries of the full music.
    boundaries = getBoundaries(idx2, duration, len(y))
    segments = getSegmentPart(boundaries, kNum, percent)
    # Cut the full file to sequences
    musicSequence = getSegmentSequence(segments, kNum, y)
    # Get the features for each part
    seqFeature = getSegmentFeature(musicSequence, kNum, fs)
    # Get each part's emotions
    seqEmotions = getSegmentEmotion(seqFeature)
    # And then choose the best of them
    mainEmotions = getMainEmotion(seqEmotions)
    # make a decision
    CueType = getCue(mainEmotions,bpm)

    CueTimeList = getCueTime(CueType,boundaries,duration,len(y))
    print(CueTimeList)