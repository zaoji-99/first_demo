import torch
import torch.nn as nn
from models.flattern import FlattenLayer
from configs import args_parser

args = args_parser()
torch.manual_seed(args.seed)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1,
                               20,
                               kernel_size=(5, 5),
                               padding=(0, 0),
                               stride=(1, 1),
                               bias=True)
        self.act1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(20,
                               50,
                               kernel_size=(5, 5),
                               padding=(0, 0),
                               stride=(1, 1),
                               bias=True)
        self.act2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = FlattenLayer()
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        out = self.maxpool1(self.act1(self.conv1(x)))
        out = self.conv2(out)
        out = self.maxpool2(self.act2(out))
        out = self.flatten(out)
        out = self.act3(self.fc1(out))
        out = self.fc2(out)
        return out


def cnn():
    return CNN()