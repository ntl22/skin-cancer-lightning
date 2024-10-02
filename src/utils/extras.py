from typing import List
import warnings
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf
import rich
import rich.syntax
import rich.tree

from .logger import RankedLogger

log = RankedLogger(name=__name__, rank_zero_only=True)


def extras(cfg: DictConfig):
    if not cfg.get("extras"):
        log.warning("No extras found in config, skipping...")
        return

    if cfg.extras.get("supress_warnings"):
        log.warning("Suppressing warnings <cfgs.extras.supress_warnings>")
        warnings.filterwarnings("ignore")

    if cfg.extras.get("print_config_tree"):
        log.warning("Printing config tree <cfgs.extras.print_config_tree>")
        print_config_tree(cfg)


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    order: List[str] = (
        "model",
        "data",
        "logger",
        "trainer",
    ),
    resolve: bool = False,
    save_to_file: bool = True,
):
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    infos = []

    for field in order:
        if field not in cfg:
            log.warning(f"Field {field} is not found, skipping...")
        else:
            infos.append(field)

    for field in cfg.keys():
        if field not in infos:
            infos.append(field)

    for field in infos:
        branch = tree.add(field, style=style)

        group = cfg[field]
        if isinstance(group, DictConfig):
            content = OmegaConf.to_yaml(group, resolve=resolve)
        else:
            content = str(group)

        branch.add(rich.syntax.Syntax(content, "yaml"))

    rich.print(tree)

    if save_to_file:
        with open(f"{cfg.paths.output_dir}/config_tree.log", "w") as f:
            rich.print(tree, file=f)
