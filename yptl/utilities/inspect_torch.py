from __future__ import annotations  # noqa: D100

import inspect
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import os.PathLike

import importlib
import sys
from pathlib import Path

import pytorch_lightning.callbacks
import torch.nn
import torch.optim

from yptl.utilities.yptldict import YPTLDict

TORCH_MODULE_IMPORTS = ["torch.nn"]
TORCH_OPTIMIZER_IMPORTS = ["torch.optim"]
TORCH_SCHEDULER_IMPORTS = ["torch.optim.lr_scheduler"]
CALLBACK_IMPORTS = ["pytorch_lightning.callbacks"]


def append_to_import_modules_list(module_name: str):
    TORCH_MODULE_IMPORTS.append(module_name)
    TORCH_OPTIMIZER_IMPORTS.append(module_name)
    TORCH_SCHEDULER_IMPORTS.append(module_name)
    CALLBACK_IMPORTS.append(module_name)


def load_sourcefile_as_module(file_path: str | os.PathLike):
    file_path = Path(file_path)
    module_name = file_path.stem

    # Import source file directly
    # see: https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Enable import also via import keyword
    # e.g. import myclass
    sys.modules[module_name] = module

    append_to_import_modules_list(module_name)


def get_cls_from_module(name: str, module: str) -> type:
    module = importlib.import_module(module)
    cls = getattr(module, name)
    if not inspect.isclass(cls):
        raise AttributeError(f"{name} must be a class, not a {cls}.")
    return cls


def get_cls_from_modules(name: str, modules: list[str]) -> type:
    for module in modules:
        try:
            return get_cls_from_module(name=name, module=module)
        except AttributeError:
            continue
    raise ClassNotFoundInModulesError(name, modules)


class ClassNotFoundInModulesError(Exception):  # noqa: D101
    def __init__(self, cls_name: str, modules: list[str]) -> None:  # noqa: D107
        available_classes = []
        for module in modules:
            available_classes += [
                cls.__name__ for _, cls in inspect.getmembers(module, inspect.isclass)
            ]
        super().__init__(
            f"Class '{cls_name}' not found in modules [{', '.join(modules)}].\nAvailable classes are: {' '.join(available_classes)}"
        )


def create_cls_from_module(name: str, module: str, args: dict | None = None):  # noqa: ANN001, ANN201, D103
    if not args:
        args = {}
    return get_cls_from_module(name, module)(**args)


def create_cls_from_modules(name: str, modules: list[str], args: dict | None = None):  # noqa: ANN001, ANN201, D103
    if not args:
        args = {}
    return get_cls_from_modules(name, modules)(**args)


def create_torch_module(name: str, args: dict | None = None) -> torch.nn.Module:  # noqa: D103
    return create_cls_from_modules(name=name, args=args, modules=TORCH_MODULE_IMPORTS)


def create_torch_module_from_yptl_dict(factory_dict: YPTLDict) -> torch.nn.Module:  # noqa: D103
    yptldict = YPTLDict(factory_dict)
    return create_torch_module(name=yptldict.type, args=yptldict.args)


def get_torch_optimizer(name: str) -> torch.optim.Optimizer:  # noqa: D103
    return get_cls_from_modules(name, TORCH_OPTIMIZER_IMPORTS)


def get_torch_lr_scheduler(name: str) -> torch.optim.lr_scheduler.LRScheduler:  # noqa: D103
    return get_cls_from_modules(name, TORCH_SCHEDULER_IMPORTS)


def create_callback(
    name: str, args: None | dict = None
) -> pytorch_lightning.callbacks.Callback:  # noqa: D103
    return create_cls_from_modules(name=name, args=args, modules=CALLBACK_IMPORTS)
