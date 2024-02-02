from __future__ import annotations  # noqa: D100

import os

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")


class MNISTDataModule(LightningDataModule):
    """
    MNIST LightningDataModule.

    Based on the example of Lightning docs
    see: https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/datamodules.html#Using-DataModules
    """

    def __init__(self, data_dir: str = PATH_DATASETS, batch_size: int = 32) -> None:
        """
        MNIST LightingDataModule.

        Args:
        ----
            data_dir (str, optional):
                Path to directory where data is located or will be downloaded into. Defaults to environment variable PATH_DATASETS.
            batch_size (int, optional):
                batch size used for DataLoader for training, validation and test data, defaults to 32.
        """
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.dims = (1, 28, 28)
        self.num_classes = 10
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        """Download data into specified directory."""
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        """
        Load in data from file and prepare PyTorch tensor datasets for each split (train, val, test).

        Args:
        ----
            stage (str, optional): fit or test stage. Defaults to None.
        """
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self) -> DataLoader:
        """Return DataLoader for training data."""
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        """Return DataLoader for validation data."""
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        """Return DataLoader for test data."""
        return DataLoader(self.mnist_test, batch_size=self.batch_size)
