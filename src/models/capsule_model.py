from typing import List
import torch
import lightning as pl

from src.models.metrics.loss import CapsNetLoss
from src.models.metrics.accuracy import CapsNetAccuracy


class CapsuleModel(pl.LightningModule):
    def __init__(
        self,
        n_classes: int,
        weights: List[float],
        backbone: torch.nn.Module,
        cbam: torch.nn.Module,
        capsule: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ):
        super().__init__()

        self.backbone = backbone
        self.cbam = cbam
        self.capsule = capsule

        self.save_hyperparameters(logger=False, ignore=["backbone", "cbam", "capsule"])

        self.loss = CapsNetLoss()
        self.criterion = CapsNetAccuracy()

    def setup(self, stage):
        if self.hparams.compile and stage == "fit":
            self.backbone = torch.compile(self.backbone)
            self.cbam = torch.compile(self.cbam)

    def forward(self, x):
        x = self.backbone(x)
        x = self.cbam(x)
        x = self.capsule(x)
        return x

    def to_one_hot(self, y):
        y_onehot = torch.zeros(y.size(0), self.hparams.n_classes, device=y.device)
        y_onehot[:, y] = 1.0
        return y_onehot

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        y_one_hot = self.to_one_hot(y)
        loss = self.loss(y_hat, y_one_hot, self.hparams.weights)
        acc = self.criterion(y_hat, y)

        self.log("train/loss", loss)
        self.log("train/acc", acc)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_one_hot = self.to_one_hot(y)

        loss = self.loss(y_hat, y_one_hot)
        acc = self.criterion(y_hat, y)

        self.log("val/loss", loss)
        self.log("val/acc", acc)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_one_hot = self.to_one_hot(y)

        loss = self.loss(y_hat, y_one_hot)
        acc = self.criterion(y_hat, y)

        self.log("test/loss", loss)
        self.log("test/acc", acc)

        return loss

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())

        if self.hparams.lr_scheduler:
            lr_scheduler = self.hparams.lr_scheduler(optimizer)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                },
            }

        return optimizer
