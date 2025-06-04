"""
Example main script to test a trained WideResNet model on the EuroSAT dataset.
"""

import torch
import torchvision
from lit_modules.custom_wide_resnet_lit import WideResnetLit
import lightning as L
from torchvision import transforms

checkpoint_path = (
    "model_training_notebooks/checkpoints/wide_resnet/epoch=49-val_acc=0.944.ckpt"
)
with open("test_indices.txt", "r") as f:
    test_indices = [int(line.strip()) for line in f.readlines()]


class DataModule(L.LightningDataModule):
    def __init__(self, test_loader):
        super().__init__()
        self.test_loader = test_loader

    def test_dataloader(self):
        return self.test_loader


def main():
    torch.set_float32_matmul_precision("high")
    L.seed_everything(42)
    model = WideResnetLit.load_from_checkpoint(checkpoint_path)
    trainer = L.Trainer(
        logger=False,
        enable_progress_bar=True,
    )
    IMAGE_SIZE = 32
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform_compose = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    dataset = torchvision.datasets.EuroSAT(
        root="model_training_notebooks/data",
        download=False,
        transform=transform_compose,
    )
    test_set = torch.utils.data.Subset(
        dataset,
        test_indices,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=64,
        shuffle=False,
        num_workers=4,
    )
    data = DataModule(test_loader)
    trainer.test(model, datamodule=data)


if __name__ == "__main__":
    main()
