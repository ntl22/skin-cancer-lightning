from typing import List
import hydra
from omegaconf import DictConfig
import rootutils
import lightning as pl

rootutils.setup_root(__file__, indicator="environment.yaml", pythonpath=True)

from lightning.pytorch.loggers import Logger
from src.utils import (
    RankedLogger,
    extras,
    task_wrapper,
    instantiate_loggers,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig):
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Training datamodule <{cfg.data._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)

    log.info(f"Instantiating loggers...")
    loggers: List[Logger] = instantiate_loggers(cfg)

    log.info("Instantiating trainer")
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=loggers,
    )

    log.info("Training...")
    trainer.fit(model, datamodule=datamodule)


@hydra.main(version_base="1.3", config_path="../config", config_name="train")
def main(cfg: DictConfig):
    extras(cfg)
    train(cfg)


if __name__ == "__main__":
    main()
