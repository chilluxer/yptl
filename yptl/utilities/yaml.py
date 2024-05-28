from __future__ import annotations  # noqa: D100

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import os
    from types import ModuleType

    from pytorch_lightning import LightningDataModule, LightningModule

import inspect
from pathlib import Path

import yaml
from pytorch_lightning import Trainer

import yptl.datamodules
import yptl.models
from yptl.utilities.inspect_torch import create_callback, get_cls_from_module
from yptl.utilities.yptldict import YPTLDict


def create_model_from_yaml_config(config: dict) -> LightningModule:
    """Create a model from yptl yaml config."""
    config = YPTLDict(config)
    return get_cls_from_module(config.type, yptl.models)(**config.args)


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
    return get_cls_from_module(config.type, yptl.datamodules)(**config.args)


class YPTLConfig:
    """Contain the configuration of the YPTL components."""

    def __init__(self, config: dict) -> None:  # noqa: D107
        config = convert_keys_to_lowercase(config)
        try:
            self._model = config.pop("model")
            self._datamodule = config.pop("datamodule")
            self._trainer = config.pop("trainer")
        except KeyError as exc:
            msg = f"YPTLConfig requires all three keywords 'model', 'datamodule' and 'trainer', but could not find keyword {exc}"
            raise KeyError(msg) from exc
        try:
            self._settings = config.pop("settings")
        except KeyError:
            self._settings = {}
        self._check()

    def __str__(self) -> str:
        """Print the YPTLConfig class in a nice string representation."""
        return yaml.dump(self._dict_repr())

    def _dict_repr(self) -> dict:
        return {
            "settings": self._settings,
            "datamodule": self._datamodule,
            "model": self._model,
            "trainer": self._trainer,
        }

    def _check(self) -> None:
        check_model_config(self._model)
        check_datamodule_config(self._datamodule)
        check_trainer_config(self._trainer)

    @property
    def model(self) -> dict:
        """Return model configuration."""
        return self._model

    @property
    def datamodule(self) -> dict:
        """Return datamodule configuration."""
        return self._datamodule

    @property
    def trainer(self) -> dict:
        """Return trainer configuration."""
        return self._trainer

    @property
    def settings(self) -> dict:
        """Return settings configuration."""
        return self._settings

    def to_yaml(self, filename: str | os.PathLike) -> None:
        """
        Write class to a YAML file.

        Args:
        ----
            dictionary (dict): The dictionary to be written.
            filename (str): The name of the output YAML file.
        """
        with Path.open(Path(filename), "w") as yaml_file:
            yaml.dump(self._dict_repr(), yaml_file)

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
            return cls(yaml.safe_load(f))


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


def check_model_config(model_config: dict) -> None:
    """Check content of the model dict of the YPTL config."""
    check_factory_and_parameters(model_config, yptl.models)


def check_datamodule_config(datamodule_config: dict) -> None:
    """Check content of the datamodule dict of the YPTL config."""
    check_factory_and_parameters(datamodule_config, yptl.datamodules)


def check_trainer_config(trainer_config: dict) -> None:
    """Check content of the trainer dict of the YPTL config."""
    check_cls_signature(Trainer, trainer_config)


def check_factory_and_parameters(config: dict, module: ModuleType) -> None:
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
