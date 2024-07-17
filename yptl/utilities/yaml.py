from __future__ import annotations  # noqa: D100

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import os

    from pytorch_lightning import LightningDataModule, LightningModule

import inspect
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pytorch_lightning import Trainer

from yptl.utilities.inspect_torch import create_callback, get_cls_from_module
from yptl.utilities.yptldict import YPTLDict


def create_model_from_yaml_config(config: dict) -> LightningModule:
    """Create a model from yptl yaml config."""
    config = YPTLDict(config)
    return get_cls_from_module(config.type, "yptl.models")(**config.args)


def create_trainer_from_config(config: dict) -> Trainer:
    """Create pytorch lightning trainer class from yptl yaml config."""
    if "callbacks" in config:
        callback_configs = (
            config["callbacks"]
            if isinstance(config["callbacks"], list)
            else [config["callbacks"]]
        )
        callback_configs = [YPTLDict(c) for c in callback_configs]
        callbacks = [create_callback(c.type, c.args) for c in callback_configs]
        config["callbacks"] = callbacks

    return Trainer(**config)


def create_datamodule_from_config(config: dict) -> LightningDataModule:
    """Create datamodule from yptl yaml config."""
    config = YPTLDict(config)
    return get_cls_from_module(config.type, "yptl.datamodules")(**config.args)


@dataclass(frozen=True)
class YPTLConfig:
    """Contain the configuration of the YPTL components."""

    model: MutableMapping
    datamodule: MutableMapping
    trainer: MutableMapping
    settings: MutableMapping = field(default_factory=dict)

    def __post_init__(self) -> None:  # noqa: D107
        for config in (self.model, self.datamodule, self.trainer):
            add_args_keyword_to_factory_dicts(config)

        check_model_config(self.model)
        check_datamodule_config(self.datamodule)
        check_trainer_config(self.trainer)

    def __str__(self) -> str:
        """Print the YPTLConfig class in a nice string representation."""
        return yaml.dump(vars(self))

    def to_yaml(self, filename: str | os.PathLike) -> None:
        """
        Write class to a YAML file.

        Args:
        ----
            dictionary (dict): The dictionary to be written.
            filename (str): The name of the output YAML file.
        """
        with Path.open(Path(filename), "w") as yaml_file:
            yaml.dump(vars(self), yaml_file)

    @classmethod
    def from_yaml(cls: YPTLConfig, filename: str | os.PathLike) -> YPTLConfig:
        """
        Create YPTLConfig from yaml file.

        Args:
        ----
            filename (str): path to yaml file

        Returns:
        -------
            YPTLConfig class object
        """
        with Path.open(Path(filename), "r") as f:
            return cls(**convert_keys_to_lowercase(yaml.safe_load(f)))


def convert_keys_to_lowercase(input_dict: dict) -> dict:
    """
    Convert all keys in the input dictionary to lowercase.

    Args:
    ----
        input_dict (dict): The dictionary whose keys need to be converted.

    Returns:
    -------
        dict: A new dictionary with lowercase keys.
    """
    return {key.lower(): value for key, value in input_dict.items()}


def add_args_keyword_to_factory_dicts(item) -> None:
    if isinstance(item, dict):
        if "type" in item:
            if "args" not in item or not item["args"]:
                item["args"] = {}
        else:
            for v in item.values():
                add_args_keyword_to_factory_dicts(v)
    elif isinstance(item, list):
        for v in item:
            add_args_keyword_to_factory_dicts(v)


def check_model_config(model_config: dict) -> None:
    """Check content of the model dict of the YPTL config."""
    check_factory_and_parameters(model_config, "yptl.models")


def check_datamodule_config(datamodule_config: dict) -> None:
    """Check content of the datamodule dict of the YPTL config."""
    check_factory_and_parameters(datamodule_config, "yptl.datamodules")


def check_trainer_config(trainer_config: dict) -> None:
    """Check content of the trainer dict of the YPTL config."""
    check_cls_signature(Trainer, trainer_config)


def check_factory_and_parameters(config: dict, module: str) -> None:
    """Check if config adheres to the yptl config rules for creating objects from dictionary entries using the type and args keywords."""
    config = YPTLDict(config)
    cls = get_cls_from_module(config.type, module)
    check_cls_signature(cls, config.args)


def check_cls_signature(cls: type, args: dict) -> None:
    """Check the class signature against a given set of arguments and prints a warning if the parameter is not used by that class."""
    signature = inspect.signature(cls)
    for k in args:
        if k not in signature.parameters:
            print(f"Warning: {k} is not an parameter of class {args.type}{signature}")  # noqa: T201
