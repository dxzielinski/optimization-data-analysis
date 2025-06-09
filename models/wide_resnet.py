import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    Basic ResNet block for CIFAR with post-activation ordering:
    conv -> BatchNorm -> ReLU -> conv -> BatchNorm -> add shortcut -> ReLU
    """
    expansion = 1  # Factor to expand channels for bottleneck blocks (unused here)

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # First convolutional layer: may change spatial size if stride != 1
        self.conv1 = nn.Conv2d(
            in_channels=in_planes,
            out_channels=planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)

        # Second convolutional layer: preserves spatial size
        self.conv2 = nn.Conv2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        # Shortcut connection: identity or projection if dims differ
        self.downsample = None
        if stride != 1 or in_planes != planes:
            # 1x1 conv to match channel dimension and spatial scale
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
        identity = x  # Preserve input for the skip connection

        # ----- First conv-BN-ReLU -----
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        # ----- Second conv-BN -----
        out = self.conv2(out)
        out = self.bn2(out)

        # If dimensions don't match, apply downsample to the identity
        if self.downsample is not None:
            identity = self.downsample(x)

        # ----- Add skip connection and final activation -----
        out += identity
        out = F.relu(out, inplace=True)
        return out


class WideResNet(nn.Module):
    """
    WideResNet for CIFAR inputs (32x32 images). Default WRN-32-4:
      - depth = 6*n + 2, default n=5 -> depth=32
      - widen_factor k=4 -> channel widths multiply base channels by k
    Architecture: conv -> {block group1, group2, group3} -> pool -> fc
    """

    def __init__(self, depth=32, widen_factor=4, num_classes=10):
        super(WideResNet, self).__init__()
        # Ensure depth matches formula
        assert (depth - 2) % 6 == 0, "Depth should be 6n+2 for CIFAR ResNet variants"
        n = (depth - 2) // 6  # Number of blocks per group

        # Initial number of channels before widening
        self.in_planes = 16

        # Initial convolution: preserves input resolution
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=self.in_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.in_planes)

        # Compute channel widths for each group
        widths = [16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        # Create three groups of blocks with increasing feature map sizes
        # Group 1: no downsampling
        self.layer1 = self._make_layer(planes=widths[0], num_blocks=n, stride=1)
        # Group 2: downsample spatially by 2 at first block
        self.layer2 = self._make_layer(planes=widths[1], num_blocks=n, stride=2)
        # Group 3: further downsample
        self.layer3 = self._make_layer(planes=widths[2], num_blocks=n, stride=2)

        # Adaptive average pooling to 1x1, then flatten + fully-connected classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(widths[2] * BasicBlock.expansion, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming initialization for ReLU activations
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                # BatchNorm weight=1, bias=0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Linear layer initialization
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, num_blocks, stride):
        """
        Create a sequential container of `num_blocks` BasicBlocks.
        The first block uses the provided stride (may downsample), others use stride=1.
        """
        # Build list of strides: first element could be >1, rest are 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            # Append a block and update in_planes for next block
            layers.append(BasicBlock(self.in_planes, planes, stride=s))
            self.in_planes = planes * BasicBlock.expansion
        # Return as a Sequential container
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial conv-BN-ReLU
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        # Pass through three block groups
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Pool to 1x1 feature map
        x = self.avgpool(x)
        # Flatten all but batch dimension
        x = torch.flatten(x, 1)
        # Final fully-connected classifier
        x = self.fc(x)
        return x
