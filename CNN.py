from torch import nn, optim

class CNN(nn.Module):
    def __init__(self):
        # 继承父类的初始化函数
        super(CNN, self).__init__()
        # 定义卷积层，1 input image channel, 25 output channels, 3*3 square convolution kernel
        # 因为MNIST数据集是黑白图像，所以input是1个channel的，即1个二维张量
        # 输入数据规模为20*32*1，其中最后一维是二维矩阵的数量，前两维是二维矩阵（方阵）的尺寸
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=3),
            nn.BatchNorm2d(25),
            nn.ReLU(inplace=True)
        )
        # ((20-3+2*0)/1)+1 = 18, ((43-3+2*0)/1)+1 = 41,18*30*25，其中最后一维由卷积核的数量决定
        # 定义池化层，max pooling over a (2, 2) window
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # ((18-2)/2)+1 = 9, ((41-2)/2)+1 = 20, 9*15*25，池化操作最后一维不变
        self.layer3 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=4),
            nn.BatchNorm2d(50),
            nn.ReLU(inplace=True)
        )
        # 输入核数25，输出核数50，((9-4+2*0)/1)+1 = 6, ((20-4+2*0)/1)+1 = 17, 6*17*50
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # ((6-2)/2)+1 = 3, ((17-2)/2)+1 = 8, 3*8*50
        self.fc = nn.Sequential(
            nn.Linear(3 * 8 * 50, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2)
        )
        # 3*6*50 -> 512 ->128 -> 32 -> 2，因为我们的目标是2个类别

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 官网上给出的展开方式，会增加一定的时间开销
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features