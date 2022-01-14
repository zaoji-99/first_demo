import torch
import matplotlib.pyplot as plt
from configs import args_parser

args = args_parser()


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def accuracy(y_hat, y):
    cmp = (y_hat.argmax(dim=1).type(y.dtype) == y).sum().float()
    return cmp


def evaluate_accuracy(model, data_iter, loss=torch.nn.CrossEntropyLoss()):
    model.eval()
    metric = Accumulator(3)
    for X, y in data_iter:
        X = X.to(args.device)
        y = y.to(args.device)
        y_pred = model(X)
        test_acc = accuracy(y_pred, y)
        test_l = loss(y_pred, y)
        metric.add(test_acc, test_l * len(y), len(y))
    model.train()
    return metric[0] / metric[2], metric[1] / metric[2]


def evaluate_poison_accuracy(model, poison_data_iter, helper, loss=torch.nn.CrossEntropyLoss()):
    model.eval()
    metric = Accumulator(3)
    for batch in poison_data_iter:
        X, y, poison_count = helper.get_poison_batch(batch, evaluation=True)
        X, y = X.to(args.device), y.to(args.device)
        y_pred = model(X)
        test_acc = accuracy(y_pred, y)
        test_l = loss(y_pred, y)
        metric.add(test_acc, test_l * poison_count, poison_count)
    model.train()
    return metric[0] / metric[2], metric[1] / metric[2]


