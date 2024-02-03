from __future__ import annotations  # noqa: D100

import inspect
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

import pytorch_lightning.callbacks
import torch.nn
import torch.optim


class ClassNotFoundInModuleError(Exception):  # noqa: D101
    def __init__(self, module: ModuleType, cls_name: str, available_modules: list) -> None:  # noqa: D107
        super().__init__(
            f"Class '{cls_name}' not found in {module.__name__}. Available classes are: {' '.join(available_modules)}"
        )


def available_classes_in_module(module: ModuleType) -> dict[str | type]:  # noqa: D103
    available_classes = [cls for _, cls in inspect.getmembers(module, inspect.isclass)]
    available_classes_names = [x.__name__ for x in available_classes]
    return dict(zip(available_classes_names, available_classes))


def create_cls_from_module(name: str, module, args: dict | None = None):  # noqa: ANN001, ANN201, D103
    if not args:
        args = {}
    return get_cls_from_module(name, module)(**args)


def get_cls_from_module(name: str, module: ModuleType):  # noqa: ANN201, D103
    available_classes = available_classes_in_module(module)
    raise_if_module_is_not_available(module, name, available_classes)

    return available_classes[name]


def create_torch_module(name: str, args: dict | None = None) -> torch.nn.Module:  # noqa: D103
    return create_cls_from_module(name=name, args=args, module=torch.nn)


def get_torch_optimizer(name: str) -> torch.optim.Optimizer:  # noqa: D103
    return get_cls_from_module(name, torch.optim)


def get_torch_lr_scheduler(name: str) -> torch.optim.lr_scheduler.LRScheduler:  # noqa: D103
    return get_cls_from_module(name, torch.optim.lr_scheduler)


def create_callback(name: str, args: None | dict = None) -> pytorch_lightning.callbacks.Callback:  # noqa: D103
    return create_cls_from_module(name=name, args=args, module=pytorch_lightning.callbacks)


def raise_if_module_is_not_available(  # noqa: D103
    module: ModuleType, cls_name: str, module_dict: dict[str, torch.nn.Module]
) -> None:
    if cls_name not in module_dict:
        raise ClassNotFoundInModuleError(module, cls_name, list(module_dict))
