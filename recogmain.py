import audioProcess
import DataProcess
import torch
import Model
import os
import numpy as np
import time


def train(pos_path, neg_path, enum = 1):
    model = Model.model(enum=enum)
    if os.path.exists('model/cnn.pth'):
        model.load_model()
    else:
        print("Warning: Do not have model.")

    if os.path.exists('audio/Preprocessed/traindata.npy') and os.path.exists('audio/Preprocessed/trainlabel.npy'):
        print("Loading data set from file.")
        Data = np.load('audio/Preprocessed/traindata.npy')
        Label = np.load('audio/Preprocessed/trainlabel.npy')
    else:
        Data, Label = DataProcess.get_traindata(pos_path, neg_path, batch_size=128)

        Data = np.array(Data)
        Label = np.array(Label)

        np.save('audio/Preprocessed/traindata.npy', Data)
        np.save('audio/Preprocessed/trainlabel.npy', Label)

    for i in range(len(Label)):
        data = Data[i]
        label = Label[i]

        data = np.array(data)
        label = np.array(label)

        data = torch.from_numpy(data)
        data = data.type(torch.FloatTensor)  # 转Float
        label = torch.from_numpy(label)
        label = label.type(torch.LongTensor)  # 转Long

        model.train_model(data, label)

        if ((i + 1) % 10 == 0):
            print("Save model in i=%d." % (i + 1))
            model.save_model()

    model.save_model()


def test(pos_path, neg_path):
    model = Model.model(enum=1)
    if os.path.exists('model/cnn_m15.pth'):
        model.load_model('model/cnn_m15.pth')
    else:
        print("Do not have model.")
        return

    if os.path.exists('audio/Preprocessed/testdata.npy') and os.path.exists('audio/Preprocessed/testlabel.npy'):
        print("Loading Data from file!")
        Data = np.load('audio/Preprocessed/testdata.npy')
        Label = np.load('audio/Preprocessed/testlabel.npy')
    else:
        Data, Label = DataProcess.get_testdata(pos_path, neg_path, batch_size=128)

        Data = np.array(Data)
        Label = np.array(Label)

        np.save('audio/Preprocessed/testdata.npy', Data)
        np.save('audio/Preprocessed/testlabel.npy', Label)

    loss = .0
    acc = .0

    for i in range(len(Label)):
        data = Data[i]
        label = Label[i]

        data = np.array(data)
        label = np.array(label)

        data = torch.from_numpy(data)
        data = data.type(torch.FloatTensor)  # 转Float
        label = torch.from_numpy(label)
        label = label.type(torch.LongTensor)  # 转Long

        curloss, curacc = model.test_model(data, label)
        loss += curloss
        acc += curacc
    loss /= len(Label)
    acc /= len(Label)
    print("Average test loss: %.6f, test accuracy: %.6f" %(loss, acc))

    
def predict(audio_path):
    Data, Label = DataProcess.get_predictdata(audio_path)
    if Data == None:
        return None

    model = Model.model(enum=1)
    if os.path.exists('model/cnn.pth'):
        model.load_model('model/cnn.pth')
    else:
        print("Error: Do not have model.")
        return

    rt_duration = []

    result = model.predict(Data[0])
    result = result[0].numpy()
        
    return result
    

if __name__ == "__main__":
    filename = input("Please input the predicted filename:")
    print(predict(filename))