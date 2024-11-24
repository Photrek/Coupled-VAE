import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from typing import Callable, Optional, Union, Sequence, List

class MyDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

class CustomCelebA(Dataset):
    """
    Custom CelebA dataset class that handles corrupted CelebA data without requiring a partition file.
    """
    def __init__(self, data_path: str, transform: Callable):
        self.data_dir = Path(data_path)
        self.transforms = transform
        self.imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])
        if self.transforms:
            img = self.transforms(img)
        return img, 0.0  # Dummy label to keep compatibility

class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module for VAE training without relying on list_eval_partition.txt.
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
        
        # Load the custom CelebA dataset without using list_eval_partition.txt
        full_dataset = CustomCelebA(self.data_dir, transform=train_transforms)
        
        # Determine lengths for train, validation, and test sets (approx. 85%, 7.5%, 7.5%)
        full_length = len(full_dataset)
        train_length = int(0.85 * full_length)
        val_length = int(0.075 * full_length)
        test_length = full_length - train_length - val_length

        # Split dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_length, val_length, test_length]
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

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=True
        )