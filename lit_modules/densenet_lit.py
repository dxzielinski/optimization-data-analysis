import torch
import torch.nn as nn
import lightning as L  # PyTorch Lightning for streamlined training
import torchmetrics  # TorchMetrics for standardized metric computation
from torchvision.models.resnet import wide_resnet50_2  # Kept for import consistency
from models.densenet import DenseNet  # Custom DenseNet implementation

# Constants defining task type and number of classes
TASK = "multiclass"
NUM_CLASSES = 10

class DenseNetLit(L.LightningModule):
    """
    LightningModule for training a custom DenseNet on a multiclass task.

    Args:
        hyperparameters (dict): Configuration dict containing:
            - depth (int): Total network depth (must satisfy 3n+4).
            - first_output (int): Number of channels after initial conv.
            - growth_rate (int): Feature growth per Dense layer.
            - dropout (float): Dropout probability for regularization.
    """
    def __init__(self, hyperparameters: dict):
        super().__init__()
        # Save provided hyperparameters for logging and checkpointing
        self.hyperparameters = hyperparameters

        # Instantiate DenseNet backbone with given settings
        self.model = DenseNet(
            depth=hyperparameters["depth"],
            first_output=hyperparameters["first_output"],
            growth_rate=hyperparameters["growth_rate"],
            dropout=hyperparameters["dropout"],
            num_classes=NUM_CLASSES,
        )

        # Define loss function for multiclass classification
        self.loss_fn = nn.CrossEntropyLoss()

        # Prepare metric collections for train/val/test phases
        metrics = torchmetrics.MetricCollection(
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
        # Clone metrics for each stage with distinct prefixes
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

        # Lists to accumulate batch-level outputs for epoch aggregation
        self.train_batch_outputs = []
        self.val_batch_outputs = []
        self.test_batch_outputs = []

    def on_train_start(self):
        """
        Called once at the beginning of training; log hyperparameters if logger is configured.
        """
        if self.logger:
            self.logger.log_hyperparams(self.hyperparameters)

    def on_test_start(self):
        """
        Called at the start of testing; repeat hyperparameter logging for consistency.
        """
        if self.logger:
            self.logger.log_hyperparams(self.hyperparameters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the DenseNet model.

        Args:
            x (Tensor): Input batch of shape (B, C, H, W).
        Returns:
            Tensor: Logits of shape (B, NUM_CLASSES).
        """
        return self.model(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Process a single training batch:
          - Compute logits, loss
          - Convert to probabilities for metric tracking
          - Store outputs for epoch-level aggregation
          - Log loss and learning rate

        Returns:
            Tensor: Loss value for backpropagation
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        # Convert logits to probabilities for metric computation
        probs = torch.softmax(logits, dim=1)
        self.train_batch_outputs.append({"probs": probs, "target": y})

        # Log instantaneous training loss
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        # Log learning rate for monitoring
        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr", current_lr, on_step=True, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        """
        At epoch end, aggregate all stored predictions and targets,
        compute metrics, log them, then reset buffers and metrics.
        """
        # Concatenate probabilities and targets across all batches
        all_probs = torch.cat([o["probs"] for o in self.train_batch_outputs])
        all_targets = torch.cat([o["target"] for o in self.train_batch_outputs])
        # Compute and log metrics
        epoch_metrics = self.train_metrics(all_probs, all_targets)
        self.log_dict(epoch_metrics)
        # Reset for next epoch
        self.train_metrics.reset()
        self.train_batch_outputs.clear()

    def validation_step(self, batch, batch_idx: int):
        """
        Process a single validation batch:
          - Compute logits, loss
          - Store outputs for metrics
          - Log validation loss
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        probs = torch.softmax(logits, dim=1)
        self.val_batch_outputs.append({"probs": probs, "target": y})
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        """
        Aggregate validation outputs, compute and log metrics, then reset.
        """
        all_probs = torch.cat([o["probs"] for o in self.val_batch_outputs])
        all_targets = torch.cat([o["target"] for o in self.val_batch_outputs])
        epoch_metrics = self.val_metrics(all_probs, all_targets)
        self.log_dict(epoch_metrics)
        self.val_metrics.reset()
        self.val_batch_outputs.clear()

    def test_step(self, batch, batch_idx: int):
        """
        Process a single test batch: store outputs for metrics (no immediate logging).
        """
        x, y = batch
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        self.test_batch_outputs.append({"probs": probs, "target": y})

    def on_test_epoch_end(self):
        """
        Aggregate test outputs, compute and log metrics, then reset.
        """
        all_probs = torch.cat([o["probs"] for o in self.test_batch_outputs])
        all_targets = torch.cat([o["target"] for o in self.test_batch_outputs])
        epoch_metrics = self.test_metrics(all_probs, all_targets)
        self.log_dict(epoch_metrics)
        self.test_metrics.reset()
        self.test_batch_outputs.clear()

    def configure_optimizers(self) -> dict:
        """
        Set up optimizer and learning rate scheduler:
          - SGD with momentum and weight decay
          - OneCycleLR for cyclical learning rate
        Returns:
            Dict describing optimizer and scheduler for Lightning
        """
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=1e-5,
        )
        # Estimate total number of training steps from trainer
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
