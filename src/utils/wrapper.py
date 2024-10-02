from importlib.util import find_spec
from typing import Callable

from omegaconf import DictConfig
from .logger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def task_wrapper(f: Callable) -> Callable:
    def wrapper(cfg: DictConfig) -> None:
        try:
            f(cfg)
        except Exception as e:
            log.exception("")
            raise e
        finally:
            if find_spec("wandb"):
                import wandb  # type: ignore

                if wandb.run is not None:
                    wandb.finish()

            log.info(f"Output dir: {cfg.paths.output_dir}")

    return wrapper
