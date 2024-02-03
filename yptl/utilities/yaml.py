from __future__ import annotations  # noqa: D100

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import os

    from pytorch_lightning import LightningDataModule, LightningModule

from pathlib import Path

import yaml
from pytorch_lightning import Trainer

import yptl
from yptl.utilities import YPTLDict
from yptl.utilities.inspect_torch import get_cls_from_module


def read_yptl_template_file(path: str | os.PathLike) -> dict:  # noqa: D103
    with Path(path).open("r") as f:
        return yaml.safe_load(f)


def create_model_from_yaml_config(config: dict) -> LightningModule:
    """Create a model from yptl yaml config."""
    config = YPTLDict(config)
    return get_cls_from_module(config.type, yptl.models)(**config.args)


def create_trainer_from_config(config: dict) -> Trainer:
    """Create pytorch lightning trainer class from yptl yaml config."""
    return Trainer(**config)


def create_datamodule_from_config(config: dict) -> LightningDataModule:
    """Create datamodule from yptl yaml config."""
    config = YPTLDict(config)
    return get_cls_from_module(config.type, yptl.datamodules)(**config.args)


def check_arguments_of_classes() -> None:  # noqa: D103
    pass
