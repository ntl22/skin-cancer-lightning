import logging
from typing import Mapping, Optional
from lightning_utilities.core.rank_zero import rank_zero_only, rank_prefixed_message


class RankedLogger(logging.LoggerAdapter):
    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Mapping[str, object] = None,
    ):
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra)
        self.rank_zero_only = rank_zero_only

    def log(self, level: int, msg: str, rank: Optional[int] = None, *args, **kwargs):
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            current_rank = getattr(rank_zero_only, "rank", None)

            if current_rank is None:
                raise RuntimeError("rank_zero_only.rank is not set.")

            msg = rank_prefixed_message(msg, current_rank)

            if self.rank_zero_only:
                if current_rank == 0:
                    self.logger.log(level, msg, *args, **kwargs)
            else:
                if current_rank == rank or rank is None:
                    self.logger.log(level, msg, *args, **kwargs)
