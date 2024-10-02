from typing import List
import hydra
from lightning.pytorch.loggers import Logger
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig

from .logger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def instantiate_loggers(cfg: DictConfig) -> List[Logger]:
    if not cfg.get("logger"):
        log.warning("No loggers found. Skipping")
        return []

    loggers = []

    for _, log_cfg in cfg.logger.items():
        if isinstance(log_cfg, DictConfig) and "_target_" in log_cfg:
            log.info(f"Instantiating logger <{log_cfg._target_}>")
            logger = hydra.utils.instantiate(log_cfg)
            loggers.append(logger)

    return loggers
