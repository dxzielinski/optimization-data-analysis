import torch
import torch.nn as nn
import lightning as L
import torchmetrics
from torchvision.models.resnet import wide_resnet50_2


TASK = "multiclass"
NUM_CLASSES = 10


class _BNReLUConv(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, dropout
    ):
        super().__init__()
        self.add_module("bn", nn.BatchNorm2d(in_channels))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
        )
        if dropout and dropout > 0:
            self.add_module("drop", nn.Dropout2d(p=dropout))


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, dropout):
        super().__init__()
        self.branch = _BNReLUConv(
            in_channels,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            dropout=dropout,
        )

    def forward(self, x):
        out = self.branch(x)
        return torch.cat([x, out], dim=1)


class Transition(nn.Sequential):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.add_module("bn", nn.BatchNorm2d(in_channels))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
        )
        if dropout and dropout > 0:
            self.add_module("drop", nn.Dropout2d(p=dropout))
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(
        self, depth=40, first_output=16, growth_rate=12, dropout=0.2, num_classes=10
    ):
        super().__init__()
        assert (depth - 4) % 3 == 0, "depth must be 3n+4"
        num_layers_per_block = (depth - 4) // 3

        self.conv1 = nn.Conv2d(
            3, first_output, kernel_size=3, stride=1, padding=1, bias=False
        )
        num_channels = first_output

        self.block1 = self._make_dense_block(
            num_layers_per_block, num_channels, growth_rate, dropout
        )
        num_channels += num_layers_per_block * growth_rate
        self.trans1 = Transition(num_channels, num_channels, dropout)

        self.block2 = self._make_dense_block(
            num_layers_per_block, num_channels, growth_rate, dropout
        )
        num_channels += num_layers_per_block * growth_rate
        self.trans2 = Transition(num_channels, num_channels, dropout)

        self.block3 = self._make_dense_block(
            num_layers_per_block, num_channels, growth_rate, dropout
        )
        num_channels += num_layers_per_block * growth_rate

        self.bn = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_channels, num_classes)

    def _make_dense_block(self, num_layers, in_channels, growth_rate, dropout):
        layers = []
        for i in range(num_layers):
            layers.append(
                DenseLayer(in_channels + i * growth_rate, growth_rate, dropout)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.trans1(x)
        x = self.block2(x)
        x = self.trans2(x)
        x = self.block3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits
