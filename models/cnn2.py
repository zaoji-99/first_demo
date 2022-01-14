import torch.nn as nn
from models.flattern import FlattenLayer


class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv2d(1,
                               32,
                               kernel_size=(5, 5),
                               padding=(0, 0),
                               stride=(1, 1),
                               bias=True)
        self.act1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(32,
                               64,
                               kernel_size=(5, 5),
                               padding=(0, 0),
                               stride=(1, 1),
                               bias=True)
        self.act2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = FlattenLayer()
        self.fc1 = nn.Linear(1024, 512)
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        out = self.maxpool1(self.act1(self.conv1(x)))
        out = self.conv2(out)
        out = self.maxpool2(self.act2(out))
        out = self.flatten(out)
        out = self.act3(self.fc1(out))
        out = self.fc2(out)
        return out


def cnn2():
    return CNN2()