import torch
import torch.nn as nn
import lightning as L  # PyTorch Lightning for structured training loops
import torchmetrics  # Standardized metric implementations
from torchvision.models.resnet import wide_resnet50_2  # Unused here, placeholder import

from models.wide_resnet import WideResNet  # Custom Wide ResNet implementation

# Task configuration constants
TASK = "multiclass"
NUM_CLASSES = 10


class WideResnetLit(L.LightningModule):
    """
    PyTorch LightningModule for training WideResNet on CIFAR-style data.

    Handles:
      - Model initialization with depth, widen factor, and classification head.
      - Loss computation (CrossEntropy).
      - Accuracy metrics for train/val/test stages.
      - Optimizer and OneCycleLR scheduler setup.
    """

    def __init__(
        self,
        depth: int = 32,
        widen_factor: int = 4,
        num_classes: int = NUM_CLASSES,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
    ):
        super().__init__()
        # Save constructor arguments to self.hparams for logging and checkpointing
        self.save_hyperparameters()

        # Instantiate WideResNet using custom implementation
        self.model = WideResNet(
            depth=depth,
            widen_factor=widen_factor,
            num_classes=num_classes,
        )
        # Loss function for multiclass classification
        self.criterion = nn.CrossEntropyLoss()

        # Accuracy metrics for different phases
        self.train_acc = torchmetrics.Accuracy(
            task=TASK, num_classes=num_classes, average="micro"
        )
        self.val_acc = torchmetrics.Accuracy(
            task=TASK, num_classes=num_classes, average="micro"
        )
        self.test_acc = torchmetrics.Accuracy(
            task=TASK, num_classes=num_classes, average="micro"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the WideResNet model.

        Args:
            x (Tensor): Input image batch of shape (B, C, H, W).
        Returns:
            Tensor: Logits of shape (B, num_classes).
        """
        return self.model(x)

    def training_step(self, batch, batch_idx: int):
        """
        Training logic for a single batch:
          - Forward pass
          - Compute loss
          - Compute and log accuracy
          - Log learning rate

        Returns:
            Tensor: training loss for backprop
        """
        inputs, targets = batch
        logits = self(inputs)
        loss = self.criterion(logits, targets)

        # Predictions and accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, targets)

        # Log training metrics to TensorBoard/progress bar
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        # Log current learning rate
        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr", current_lr, on_step=True, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx: int):
        """
        Validation logic for a single batch:
          - Forward pass
          - Compute loss
          - Compute and log accuracy
        """
        inputs, targets = batch
        logits = self(inputs)
        loss = self.criterion(logits, targets)

        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, targets)

        # Log validation metrics (epoch-level)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx: int):
        """
        Test logic for a single batch:
          - Forward pass
          - Compute loss
          - Compute and log accuracy
        """
        inputs, targets = batch
        logits = self(inputs)
        loss = self.criterion(logits, targets)

        preds = torch.argmax(logits, dim=1)
        acc = self.test_acc(preds, targets)

        # Log test metrics (epoch-level)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """
        Define optimizer and learning rate scheduler:
          - SGD with momentum and weight decay
          - OneCycleLR scheduler for super-convergence
        Returns dict for Lightning.
        """
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        # Estimate total training steps for the scheduler
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=1.0,
            total_steps=total_steps,
            div_factor=10,
            final_div_factor=23,
        )
        # Return Lightning-formatted optimizer and scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "strict": False,
            },
        }
