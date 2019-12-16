import audioProcess
from random import shuffle
import torch
import numpy as np


def getfilename(file_dir):
    if os.path.isfile(file_dir):
        return [file_dir]

    filelist = []
    for root, dirs, files in os.walk(file_dir):
        #print("root = ", root)
        #print("dirs = ", dirs)
        #print("files = ", files)
        for file in files:
            filelist.append(root + '/' + file)

        if len(dirs) > 0:
            for dir in dirs:
                dirfiles = getfilename(root + '/' + dir)
                filelist.extend(dirfiles)
        break

    return filelist


def get_traindata(pos_path, neg_path, batch_size = 128):
    pos_files = getfilename(pos_path)
    neg_files = getfilename(neg_path)

    files = pos_files + neg_files
    shuffle(files)

    X = []
    y = []

    for i, file in enumerate(files):
        print("----Processing the %d file:%s----" % (i + 1, file))
        if file in pos_files:
            mfccs = audioProcess.get_feature(file)
            data, label = audioProcess.process_mfccs(mfccs, type=1, feature_len=43)
        else:
            mfccs = audioProcess.get_feature(file)
            data, label = audioProcess.process_mfccs(mfccs, type=0, feature_len=43)

        # 音频长度短于阈值，舍弃之
        if len(label) <= 0:
            print("Drop file %s cause length of the voice is too short!" % file)
            continue

        X.extend(data)
        y.extend(label)

    print("Loading finish, divide into batches")

    index = list(range(len(y)))
    shuffle(index)

    Data = []
    Label = []

    k = int(len(y)/batch_size)

    for i in range(k):
        # 获取index中batch_size个索引，生成一批训练数据
        curdata = []
        curlabel = []
        for j in range(batch_size*i, batch_size*(i+1)):
            curdata.append([X[index[j]]])
            curlabel.append(y[index[j]])
        Data.append(curdata)
        Label.append(curlabel)

    print("Successful generate %d batch data for train!" %k)

    return Data, Label


def get_testdata(pos_path, neg_path, batch_size=128):
    pos_files = getfilename(pos_path)
    neg_files = getfilename(neg_path)

    files = pos_files + neg_files

    X = []
    y = []

    for i, file in enumerate(files):
        print("----Processing the %d file:%s----" % (i + 1, file))
        if file in pos_files:
            mfccs = audioProcess.get_feature(file)
            data, label = audioProcess.process_mfccs(mfccs, type=1, feature_len=43)
        else:
            mfccs = audioProcess.get_feature(file)
            data, label = audioProcess.process_mfccs(mfccs, type=0, feature_len=43)

        # 音频长度短于阈值，舍弃之
        if len(label) <= 0:
            print("Drop file %s cause length of the voice is too short!" % file)
            continue

        X.extend(data)
        y.extend(label)

    print("Loading finish, divide into batches")

    Data = []
    Label = []

    k = int(len(y) / batch_size)

    for i in range(k):
        # 获取index中batch_size个索引，生成一批训练数据
        curdata = []
        curlabel = []
        for j in range(batch_size * i, batch_size * (i + 1)):
            curdata.append([X[j]])
            curlabel.append(y[j])
        Data.append(curdata)
        Label.append(curlabel)

    print("Successful generate %d batch data for test!" % k)

    return Data, Label


def get_predictdata(audio_path):
    file = audio_path
    
    Data = []
    Label = []
    
    mfccs = audioProcess.get_feature(file)
    data, label = audioProcess.process_mfccs(mfccs, type=1, feature_len=43)

    # 音频长度短于阈值，舍弃之
    if len(label) <= 0:
        return None, None

    # 规范化特征
    for i in range(label.shape[0]):
        data[i] = [data[i]]
    data = np.array(data)

    data = torch.from_numpy(data)
    data = data.type(torch.FloatTensor)
    label = torch.from_numpy(label)
    label = label.type(torch.LongTensor)

    Data.append(data)
    Label.append(label)

    return Data, Label