import torch
import torch.nn as nn
import lightning as L
import torchmetrics
from torchvision.models.resnet import wide_resnet50_2  # (not used here, but kept for consistency)
from models.densenet import DenseNet

TASK = "multiclass"
NUM_CLASSES = 10

class DenseNetLit(L.LightningModule):
    def __init__(self, hyperparameters):
        """
        Example hyperparameters:
        depth=40, first_output=16, growth_rate=12, dropout=0.2
        hyperparameters = {
            "depth": 40,
            "first_output": 16,
            "growth_rate": 12,
            "dropout": 0.2,
        }
        """
        super().__init__()
        self.model = DenseNet(
            hyperparameters["depth"],
            hyperparameters["first_output"],
            hyperparameters["growth_rate"],
            hyperparameters["dropout"],
            num_classes=NUM_CLASSES,
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.train_metrics = torchmetrics.MetricCollection(
            {
                "f1_macro": torchmetrics.F1Score(
                    task=TASK, num_classes=NUM_CLASSES, average="macro"
                ),
                "precision": torchmetrics.Precision(
                    task=TASK, num_classes=NUM_CLASSES, average="macro"
                ),
                "recall": torchmetrics.Recall(
                    task=TASK, num_classes=NUM_CLASSES, average="macro"
                ),
                "auroc": torchmetrics.AUROC(
                    task=TASK, num_classes=NUM_CLASSES, average="macro"
                ),
                "accuracy": torchmetrics.Accuracy(task=TASK, num_classes=NUM_CLASSES),
            },
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")
        self.train_batch_outputs = []
        self.val_batch_outputs = []
        self.test_batch_outputs = []
        self.hyperparameters = hyperparameters

    def on_train_start(self):
        if self.logger is not None:
            self.logger.log_hyperparams(self.hyperparameters)

    def on_test_start(self):
        if self.logger is not None:
            self.logger.log_hyperparams(self.hyperparameters)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        probabilities = torch.softmax(logits, dim=1)
        self.train_batch_outputs.append({"probabilities": probabilities, "y": y})
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log(
            "lr", self.optimizers().param_groups[0]["lr"], on_step=True, on_epoch=True
        )
        return loss

    def on_train_epoch_end(self):
        probabilities = torch.cat(
            [x["probabilities"] for x in self.train_batch_outputs]
        )
        y = torch.cat([x["y"] for x in self.train_batch_outputs])
        metrics = self.train_metrics(probabilities, y)
        self.log_dict(metrics)
        self.train_metrics.reset()
        self.train_batch_outputs.clear()

    def validation_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        probabilities = torch.softmax(logits, dim=1)
        self.val_batch_outputs.append({"probabilities": probabilities, "y": y})
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        probabilities = torch.cat([x["probabilities"] for x in self.val_batch_outputs])
        y = torch.cat([x["y"] for x in self.val_batch_outputs])
        metrics = self.val_metrics(probabilities, y)
        self.log_dict(metrics)
        self.val_metrics.reset()
        self.val_batch_outputs.clear()

    def test_step(self, batch):
        x, y = batch
        logits = self(x)
        probabilities = torch.softmax(logits, dim=1)
        self.test_batch_outputs.append({"probabilities": probabilities, "y": y})

    def on_test_epoch_end(self):
        probabilities = torch.cat([x["probabilities"] for x in self.test_batch_outputs])
        y = torch.cat([x["y"] for x in self.test_batch_outputs])
        metrics = self.test_metrics(probabilities, y)
        self.log_dict(metrics)
        self.test_metrics.reset()
        self.test_batch_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.1, weight_decay=10e-6, momentum=0.9
        )
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=4.0,
            total_steps=total_steps,
            div_factor=40,
            final_div_factor=40,
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

