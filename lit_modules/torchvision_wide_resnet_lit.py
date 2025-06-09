import torch
import torch.nn as nn
import lightning as L  # PyTorch Lightning for organizing training loops
import torchmetrics  # TorchMetrics for calculating metrics in a Lightning-friendly way
from torchvision.models.resnet import wide_resnet50_2  # Predefined Wide ResNet backbone

# Task configuration
TASK = "multiclass"
NUM_CLASSES = 10


class WideResnetLit(L.LightningModule):
    """
    LightningModule wrapping a Wide ResNet-50-2 for multiclass classification.

    Handles model construction, optimization, loss computation, metric logging,
    and learning rate scheduling (OneCycleLR).
    """

    def __init__(self, hyperparameters):
        super().__init__()
        # Save hyperparameters (for logging and checkpointing)
        self.hyperparameters = hyperparameters

        # Model: Wide ResNet-50-2 with final fc adapted to NUM_CLASSES
        self.model = wide_resnet50_2(num_classes=NUM_CLASSES)

        # Loss function: standard cross-entropy for multiclass
        self.loss_fn = nn.CrossEntropyLoss()

        # Setup metric collections for train/val/test
        # Using macro-averaging for F1, precision, recall, AUROC; accuracy unaveraged
        base_metrics = torchmetrics.MetricCollection(
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
                "accuracy": torchmetrics.Accuracy(
                    task=TASK, num_classes=NUM_CLASSES
                ),
            }
        )
        # Clone for each stage with distinct prefixes
        self.train_metrics = base_metrics.clone(prefix="train_")
        self.val_metrics = base_metrics.clone(prefix="val_")
        self.test_metrics = base_metrics.clone(prefix="test_")

        # Buffers to accumulate batch outputs for epoch-level metric computation
        self.train_batch_outputs = []
        self.val_batch_outputs = []
        self.test_batch_outputs = []

    def configure_optimizers(self):
        """
        Define optimizer (SGD with momentum) and OneCycleLR scheduler.
        Scheduler steps on every training step.
        """
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=1e-3,
        )
        # Estimate total training steps for OneCycleLR
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
                "interval": "step",  # update LR each step
                "frequency": 1,
                "strict": False,      # allow scheduler even if not enough steps
            },
        }

    def on_train_start(self):
        """
        Log hyperparameters once at the beginning of training.
        """
        if self.logger:
            self.logger.log_hyperparams(self.hyperparameters)

    def on_test_start(self):
        """
        Log hyperparameters at the start of testing (for consistency).
        """
        if self.logger:
            self.logger.log_hyperparams(self.hyperparameters)

    def forward(self, x):
        """
        Forward pass through the underlying Wide ResNet model.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx=None):
        """
        Process a single training batch: compute loss, store probabilities for metrics,
        and log loss and learning rate.
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        # Convert logits to probabilities for metric calculation
        probs = torch.softmax(logits, dim=1)
        # Accumulate outputs
        self.train_batch_outputs.append({"probs": probs, "y": y})

        # Log training loss and current LR
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr", current_lr, on_step=True, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        """
        At epoch end, compute and log aggregated metrics over all training batches.
        Then clear buffers for the next epoch.
        """
        probs = torch.cat([o["probs"] for o in self.train_batch_outputs])
        targets = torch.cat([o["y"] for o in self.train_batch_outputs])
        metrics = self.train_metrics(probs, targets)
        self.log_dict(metrics)
        # Reset metrics and buffers
        self.train_metrics.reset()
        self.train_batch_outputs.clear()

    def validation_step(self, batch, batch_idx=None):
        """
        Validation batch: compute loss and store probabilities for epoch metrics.
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        probs = torch.softmax(logits, dim=1)
        self.val_batch_outputs.append({"probs": probs, "y": y})
        # Log validation loss
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        """
        Compute and log aggregated validation metrics, then reset.
        """
        probs = torch.cat([o["probs"] for o in self.val_batch_outputs])
        targets = torch.cat([o["y"] for o in self.val_batch_outputs])
        metrics = self.val_metrics(probs, targets)
        self.log_dict(metrics)
        self.val_metrics.reset()
        self.val_batch_outputs.clear()

    def test_step(self, batch, batch_idx=None):
        """
        Test batch: store probabilities for final metrics computation.
        No loss logging by default.
        """
        x, y = batch
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        self.test_batch_outputs.append({"probs": probs, "y": y})

    def on_test_epoch_end(self):
        """
        Compute and log aggregated test metrics, then reset buffers.
        """
        probs = torch.cat([o["probs"] for o in self.test_batch_outputs])
        targets = torch.cat([o["y"] for o in self.test_batch_outputs])
        metrics = self.test_metrics(probs, targets)
        self.log_dict(metrics)
        self.test_metrics.reset()
        self.test_batch_outputs.clear()
