import torch
import torch.nn as nn
import lightning as L  # PyTorch Lightning (not used directly in this module)
import torchmetrics  # TorchMetrics for evaluating during training (not used here)
from torchvision.models.resnet import wide_resnet50_2  # Imported for potential backbone usage

# Constants for task type and number of classes
TASK = "multiclass"
NUM_CLASSES = 10


class _BNReLUConv(nn.Sequential):
    """
    A helper sequential block: BatchNorm -> ReLU -> Conv2d -> (optional) Dropout

    Args:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output feature channels.
        kernel_size (int or tuple): Convolution kernel size.
        stride (int or tuple): Convolution stride.
        padding (int or tuple): Convolution padding.
        dropout (float): Dropout probability (0 disables dropout).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout):
        super().__init__()
        # Apply Batch Normalization to input channels
        self.add_module("bn", nn.BatchNorm2d(in_channels))
        # Apply ReLU activation
        self.add_module("relu", nn.ReLU(inplace=True))
        # Apply convolution (no bias since BN follows)
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
        # Optional spatial dropout for regularization
        if dropout and dropout > 0:
            self.add_module("drop", nn.Dropout2d(p=dropout))


class DenseLayer(nn.Module):
    """
    Single layer within a DenseBlock that appends its output to the input feature maps.

    Args:
        in_channels (int): Number of input feature channels.
        growth_rate (int): Number of new feature maps produced by this layer.
        dropout (float): Dropout probability after convolution.
    """
    def __init__(self, in_channels, growth_rate, dropout):
        super().__init__()
        # Branch: BN -> ReLU -> Conv to produce `growth_rate` features
        self.branch = _BNReLUConv(
            in_channels,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            dropout=dropout,
        )

    def forward(self, x):
        # Compute new features
        out = self.branch(x)
        # Concatenate input and new features (dense connectivity)
        return torch.cat([x, out], dim=1)


class Transition(nn.Sequential):
    """
    Transition layer between DenseBlocks: compresses channels and downsamples spatial dims.

    Args:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output feature channels (compression).
        dropout (float): Dropout probability after convolution.
    """
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        # BatchNorm + ReLU
        self.add_module("bn", nn.BatchNorm2d(in_channels))
        self.add_module("relu", nn.ReLU(inplace=True))
        # 1x1 convolution for channel compression
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
        # Optional dropout
        if dropout and dropout > 0:
            self.add_module("drop", nn.Dropout2d(p=dropout))
        # Average pooling to halve spatial dimensions
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """
    DenseNet implementation for CIFAR-like tasks.

    Architecture:
      1) Initial conv layer
      2) Three DenseBlocks, each followed by a Transition (except after last block)
      3) Final BN-ReLU, global average pooling, and fully-connected classifier

    Args:
        depth (int): Total network depth (must satisfy depth = 3n + 4).
        first_output (int): Number of channels after initial conv.
        growth_rate (int): Feature-map increase per DenseLayer.
        dropout (float): Dropout probability in layers.
        num_classes (int): Number of output classes.
    """
    def __init__(
        self,
        depth=40,
        first_output=16,
        growth_rate=12,
        dropout=0.2,
        num_classes=10,
    ):
        super().__init__()
        # Ensure valid depth
        assert (depth - 4) % 3 == 0, "depth must be 3n+4"
        # Number of layers per DenseBlock
        num_layers_per_block = (depth - 4) // 3

        # Initial convolution: preserve 32x32 resolution
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=first_output,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        num_channels = first_output  # Track current channel count

        # --- Dense Block 1 ---
        self.block1 = self._make_dense_block(
            num_layers_per_block,
            num_channels,
            growth_rate,
            dropout,
        )
        # After block, channels increase by growth_rate * layers
        num_channels += num_layers_per_block * growth_rate
        # Transition 1: compress + downsample
        self.trans1 = Transition(num_channels, num_channels, dropout)

        # --- Dense Block 2 ---
        self.block2 = self._make_dense_block(
            num_layers_per_block,
            num_channels,
            growth_rate,
            dropout,
        )
        num_channels += num_layers_per_block * growth_rate
        # Transition 2
        self.trans2 = Transition(num_channels, num_channels, dropout)

        # --- Dense Block 3 (no transition after) ---
        self.block3 = self._make_dense_block(
            num_layers_per_block,
            num_channels,
            growth_rate,
            dropout,
        )
        num_channels += num_layers_per_block * growth_rate

        # Final normalization and pooling/classification head
        self.bn = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_channels, num_classes)

    def _make_dense_block(self, num_layers, in_channels, growth_rate, dropout):
        """
        Construct a DenseBlock with `num_layers` DenseLayer modules.

        Each layer receives all previous feature maps via concatenation.
        """
        layers = []
        for i in range(num_layers):
            layers.append(
                DenseLayer(
                    in_channels + i * growth_rate,
                    growth_rate,
                    dropout,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial conv
        x = self.conv1(x)

        # Pass through DenseBlock + Transition sequences
        x = self.block1(x)
        x = self.trans1(x)
        x = self.block2(x)
        x = self.trans2(x)
        x = self.block3(x)

        # Final BN-ReLU
        x = self.bn(x)
        x = self.relu(x)

        # Global average pooling to 1x1
        x = self.avgpool(x)
        # Flatten to (batch_size, num_channels)
        x = torch.flatten(x, 1)
        # Classifier
        logits = self.fc(x)
        return logits
