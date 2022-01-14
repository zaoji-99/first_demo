from models.cnn import cnn
from models.cnn2 import cnn2
from models.vgg import vgg16, vgg19
from models.resnet import resnet18
from configs import args_parser
import torch.nn as nn
import torchvision.models as tmodels

args = args_parser()


def weights_init(net):
    for layer in net.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)


def create_model(dataset_name, model_name):
    model = None
    if dataset_name == 'mnist' and model_name == 'cnn':
        model = cnn()
    elif dataset_name == 'fmnist' and model_name == 'cnn':
        model = cnn2()
    elif dataset_name == 'cifar10':
        if model_name == 'vgg16':
            model = vgg16()
        elif model_name == 'resnet18':
            model = resnet18()
    elif dataset_name == 'cifar100':
        if model_name == 'vgg19':
            model = vgg19()
    elif dataset_name == 'tinyimagenet':
        if model_name == 'resnet18':
            model = tmodels.resnet18()
            model.avgpool = nn.AdaptiveAvgPool2d(1)
            model.fc = nn.Linear(model.fc.in_features, 200)
    return model.to(args.device)


# from torchstat import stat
# model = create_model(args.dataset, args.model_name)
# print(model)
# stat(model, (1, 28, 28))
