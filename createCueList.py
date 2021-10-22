import data
import random
# 0舒服	1大气	2孤独	3快乐	4梦幻	5伤感	6温暖	7清新

def getCue(mainEmotions,bpm):
    Cue = []
    bpm = bpm/3
    for item in mainEmotions:
        wave = random.choice(data.waveform[item[0]][item[1]])
        color = random.choice(data.color[item[0]][item[1]])
        Cue.append([wave, color,bpm])
    return Cue

def getCueTime(Cuetype,boundaries,duration,ly):
    # 开始，结束，分段类型，波形，颜色
    factor = duration/ly

    for item in boundaries:
        item[0] = item[0] * factor
        item[1] = item[1] * factor
        item.append(Cuetype[item[2]][0])
        item.append(Cuetype[item[2]][1])
        item.append(Cuetype[item[2]][2])
        # print (item)

    return boundaries
