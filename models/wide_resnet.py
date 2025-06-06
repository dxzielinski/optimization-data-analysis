import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import torchmetrics


class BasicBlock(nn.Module):
    """Basic ResNet block for CIFAR with post-activation ordering (conv → BN → ReLU)."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_planes,
            out_channels=planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_planes,
                    out_channels=planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out, inplace=True)
        return out


class WideResNet(nn.Module):
    """
    WideResNet for CIFAR-style inputs. Implements WRN-32-k by default:
    - depth = 6n + 2, here n = 5 → depth = 32
    - widen_factor = k = 4 → widths [16*k, 32*k, 64*k] in the three groups
    """

    def __init__(self, depth=32, widen_factor=4, num_classes=10):
        super(WideResNet, self).__init__()
        assert (depth - 2) % 6 == 0, "Depth should be 6n+2 for CIFAR ResNet variants"
        n = (depth - 2) // 6

        self.in_planes = 16

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=self.in_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.in_planes)

        widths = [16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        self.layer1 = self._make_layer(planes=widths[0], num_blocks=n, stride=1)
        self.layer2 = self._make_layer(planes=widths[1], num_blocks=n, stride=2)
        self.layer3 = self._make_layer(planes=widths[2], num_blocks=n, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(widths[2] * BasicBlock.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, num_blocks, stride):
        """
        Create one group (layer) consisting of `num_blocks` BasicBlocks.
        The first block may downsample with stride > 1; subsequent blocks use stride=1.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride=s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
