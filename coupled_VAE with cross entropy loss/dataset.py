import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
from pytorch_lightning import LightningDataModule
from pathlib import Path
from torchvision.datasets.folder import default_loader
from typing import Callable, Optional, Sequence, Union, List

class MyDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

class MyCelebA(CelebA):
    """
    Custom CelebA dataset class to handle specific needs.
    """

    def _check_integrity(self) -> bool:
        return True

class OxfordPets(Dataset):
    """
    Custom dataset for OxfordPets, replaceable with your own dataset.
    """
    def __init__(self, data_path: str, split: str, transform: Callable, **kwargs):
        self.data_dir = Path(data_path)
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])
        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])
        if self.transforms is not None:
            img = self.transforms(img)
        return img, 0.0  # Return dummy label to prevent issues

class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module for VAE training.
    """
    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(148),
            transforms.Resize(self.patch_size),
            transforms.ToTensor()
        ])
        
        val_transforms = transforms.Compose([
            transforms.CenterCrop(148),
            transforms.Resize(self.patch_size),
            transforms.ToTensor()
        ])

        # Load train and validation datasets
        self.train_dataset = MyCelebA(
            self.data_dir,
            split='train',
            transform=train_transforms,
            download=False,
        )
        
        self.val_dataset = MyCelebA(
            self.data_dir,
            split='test',
            transform=val_transforms,
            download=False,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            persistent_workers=True
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=True
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            persistent_workers=True
        )