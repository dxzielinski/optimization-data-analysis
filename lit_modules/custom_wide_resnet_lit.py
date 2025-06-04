import torch
import torch.nn as nn
import lightning as L
import torchmetrics
from torchvision.models.resnet import wide_resnet50_2

from models.wide_resnet import WideResNet

TASK = "multiclass"
NUM_CLASSES = 10


class WideResnetLit(L.LightningModule):
    """
    PyTorch Lightning module wrapping the WideResNet model for CIFAR-10.
    Includes training/validation/test steps, optimizer, and LR scheduler.
    """

    def __init__(
        self,
        depth: int = 32,
        widen_factor: int = 4,
        num_classes: int = 10,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
    ):
        super(WideResnetLit, self).__init__()
        self.save_hyperparameters()

        self.model = WideResNet(
            depth=depth,
            widen_factor=widen_factor,
            num_classes=num_classes,
        )
        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)
        loss = self.criterion(logits, targets)

        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, targets)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "lr", self.optimizers().param_groups[0]["lr"], on_step=True, on_epoch=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)
        loss = self.criterion(logits, targets)

        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, targets)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)
        loss = self.criterion(logits, targets)

        preds = torch.argmax(logits, dim=1)
        acc = self.test_acc(preds, targets)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            momentum=self.hparams.momentum,
        )
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=1.0,
            total_steps=total_steps,
            div_factor=10,
            final_div_factor=23,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "strict": False,
            },
        }
