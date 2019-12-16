from __future__ import print_function
import CNN
import torch
from torch import nn, optim
from torch.autograd import Variable
import tensorflow as tf

class model():
    def __init__(self, learning_rate=0.01, enum=1):
        # ----预定义参数----
        self.learning_rate = learning_rate
        self.enum = enum
        self.available_gpu = torch.cuda.is_available()
        self.available_gpu = False
        # ----选择模型----
        self.model = CNN.CNN()
        if self.available_gpu:
            self.model = self.model.cuda()
        # ----定义损失函数和优化器----
        self.criterion = nn.CrossEntropyLoss()  # 交叉熵
        # self.criterion = nn.MSELoss() # 均值误差，在此输出不适用
        # 优化器自动更新model中各层的参数，存储在parameters中；lr是学习率
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        # self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.learning_rate)
        self.optimizer = optim.Adam(self.model.parameters())
        # self.optimizer = optim.ASGD(self.model.parameters())


    def train_model(self, train_data, train_label, maxloss=1e-3):
        running_acc = .0
        running_loss = .0
        for epoch in range(self.enum):
            running_acc = .0
            running_loss = .0

            img = train_data
            label = train_label
            # show images，显示一个batch的图像集，很有意思
            # imshow(torchvision.utils.make_grid(img))

            if self.available_gpu:
                img = img.cuda()
                label = label.cuda()
            else:
                img = Variable(img)
                label = Variable(label)

            # 需要清除已存在的梯度,否则梯度将被累加到已存在的梯度
            self.optimizer.zero_grad()
            # forward
            out = self.model(img)
            # backward
            loss = self.criterion(out, label)
            loss.backward()
            # 更新参数optimize
            self.optimizer.step()

            running_loss += loss.item() * label.size(0)

            # 按维度dim返回最大值及索引
            _, pred = torch.max(out, dim=1)
            current_num = (pred == label).sum()
            running_acc += current_num.item()


            acc = (pred == label).float().mean()
            print("epoch: {}/{}, loss: {:.6f}, running_acc: {:.6f}".format(epoch + 1, self.enum,
                                                                                         loss.item(), acc.item()))

            if loss.item() < maxloss:
                print("Warning: training break: current loss has less than maxloss!")
                break

        return running_loss / len(train_label), running_acc / len(train_label)


    def test_model(self, test_data, test_label):
        # 指示model进入测试模式
        self.model.eval()
        eval_loss = .0
        eval_acc = .0

        img = test_data
        label = test_label

        if self.available_gpu:
            img = img.cuda()
            label = label.cuda()
        else:
            img = Variable(img)
            label = Variable(label)

        out = self.model(img)
        loss = self.criterion(out, label)

        # 这里与训练不同的是，仅计算损失，不反向传播误差和计算梯度
        eval_loss += loss.data.item() * label.size(0)
        _, pred = torch.max(out, dim=1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()

        print('**testing loss: {:.6f}, testing accuracy: {:.6f}**'.format(
            eval_loss / (len(test_label)),
            eval_acc / (len(test_label))
        ))

        return eval_loss / (len(test_label)), eval_acc / (len(test_label))


    def predict(self, X):
        # 指示model进入测试模式
        self.model.eval()
        y = []

        img = X

        if self.available_gpu:
            img = img.cuda()
        else:
            img = Variable(img)

        out = self.model(img)
        _, pred = torch.max(out, dim=1)

        y.append(pred)
        return y


    def save_model(self, path = 'model/cnn.pth'):
        torch.save(self.model.state_dict(), path)  # 保存模型


    def load_model(self, path = 'model/cnn.pth'):
        self.model.load_state_dict(torch.load(path))