from pathlib import Path
import os
import shutil
import pandas as pd
import lightning as pl
import numpy as np
from monai.apps import download_and_extract
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import albumentations as A
import albumentations.pytorch as album_pytorch

"""
Source: https://discuss.pytorch.org/t/integrating-albumentations-with-torchvision-returns-keyerror/161025
"""


class Transforms:
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))["image"]


class ISIC2018DataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir: str, batch_size: int, num_workers: int, pin_memory: bool
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.batch_size_per_device = batch_size

    def download_data(self, images: str, labels: str):
        url = "https://isic-challenge-data.s3.amazonaws.com/2018/{}.zip"

        download_and_extract(
            url.format(images), file_type="zip", output_dir=self.data_dir
        )
        download_and_extract(
            url.format(labels), file_type="zip", output_dir=self.data_dir
        )

    def convert_to_image_folder(self, images_dir: Path, labels_dir: Path, type: str):
        if os.path.exists(self.data_dir / type):
            shutil.rmtree(self.data_dir / type)

        os.makedirs(self.data_dir / type)

        df = str(labels_dir / f"{os.path.basename(labels_dir)}.csv")
        df = pd.read_csv(df)

        lesions = df.columns[1:].to_list()
        for lesion in lesions:
            os.makedirs(self.data_dir / type / lesion.lower())

        images = list(df["image"])
        df.set_index("image", inplace=True)

        for image in images:
            for lesion in lesions:
                label = df.loc[image, lesion]
                if label == 1:
                    shutil.move(
                        images_dir / f"{image}.jpg",
                        self.data_dir / type / lesion.lower() / f"{image}.jpg",
                    )

        shutil.rmtree(images_dir)
        shutil.rmtree(labels_dir)

    def prepare_data(self):
        root_dir = Path(self.data_dir)
        train_dir = root_dir / "train"
        val_dir = root_dir / "val"

        file_name = {
            "train": {
                "images": "ISIC2018_Task3_Training_Input",
                "labels": "ISIC2018_Task3_Training_GroundTruth",
            },
            "val": {
                "images": "ISIC2018_Task3_Validation_Input",
                "labels": "ISIC2018_Task3_Validation_GroundTruth",
            },
        }

        # Train dataset
        if not os.path.exists(train_dir):
            self.download_data(
                file_name["train"]["images"], file_name["train"]["labels"]
            )
            self.convert_to_image_folder(
                root_dir / file_name["train"]["images"],
                root_dir / file_name["train"]["labels"],
                "train",
            )

        # Validation dataset
        if not os.path.exists(val_dir):
            self.download_data(file_name["val"]["images"], file_name["val"]["labels"])
            self.convert_to_image_folder(
                root_dir / file_name["val"]["images"],
                root_dir / file_name["val"]["labels"],
                "val",
            )

    def num_classes(self):
        return len(os.listdir(self.data_dir / "train"))

    def setup(self, stage=None):
        if self.trainer:
            if self.batch_size_per_device % self.trainer.world_size != 0:
                raise ValueError("Batch size should be divisible by the number of GPUs")
            self.batch_size_per_device = (
                self.batch_size_per_device // self.trainer.world_size
            )

        train_transform = A.Compose(
            [
                A.Resize(299, 299),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5
                ),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(),
                album_pytorch.ToTensorV2(),
            ]
        )

        val_transform = A.Compose(
            [
                A.Resize(299, 299),
                A.Normalize(),
                album_pytorch.ToTensorV2(),
            ]
        )

        self.train_dataset = ImageFolder(
            self.data_dir / "train",
            transform=Transforms(train_transform),
        )
        self.val_dataset = ImageFolder(
            self.data_dir / "val",
            transform=Transforms(val_transform),
        )

    def train_dataloader(self):
        class_count = np.bincount(self.train_dataset.targets)
        weight = np.sum(class_count) / (self.num_classes() * class_count)
        # Round to 3 decimal places
        weight = np.round(weight, 3)
        print(f"[{', '.join(weight.astype(str))}]")
        exit()

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
