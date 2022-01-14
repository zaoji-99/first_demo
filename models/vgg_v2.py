import torch.nn as nn
from models.flattern import FlattenLayer


cfg = {
    'VGG11s': (32, 'M', 64, 'M', 128, 128, 'M', 128, 128, 'M', 128, 128, 'M'),
    'VGG11': (64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'),
    'VGG13': (64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'),
    'VGG16': (64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'),
    'VGG19': (64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'),
}


class VGG(nn.Module):
    def __init__(self, config, name, size, out=10):
        super(VGG, self).__init__()
        self.config = config
        self.model = self._make_layers(self.config, name, size, out)

    def forward(self, x):
        out = self.model(x)
        return out

    def _make_layers(self, cfg, name, size, out):
        layers = []
        in_channels = 3
        for i, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=(3, 3), padding=(1, 1)),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
                in_channels = x
        if name == 'VGG16':
            layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        layers += [FlattenLayer()]
        if name == 'VGG16':
            layers += [nn.Linear(size, out)]
        elif name == 'VGG11s':
            layers += [nn.Linear(size, size)]
            layers += [nn.ReLU()]
            layers += [nn.Linear(size, size)]
            layers += [nn.ReLU()]
            layers += [nn.Linear(size, out)]
        return nn.Sequential(*layers)


def vgg11s():
    return VGG(cfg['VGG11s'], name='VGG11s', size=128)


def vgg16():
    return VGG(cfg['VGG16'], name='VGG16', size=512)
