from __future__ import annotations  # noqa: D100

import inspect

import torch.nn
import torch.optim


class ClassNotFoundInModuleError(Exception):  # noqa: D101
    def __init__(self, module_name: str, available_modules: list) -> None:  # noqa: D107
        super().__init__(f"{module_name} not found in list of available modules: {' '.join(available_modules)}")


def available_classes_in_module(module) -> dict[str | type]:  # noqa: ANN001, D103
    available_classes = [cls for _, cls in inspect.getmembers(module, inspect.isclass)]
    available_classes_names = [x.__name__ for x in available_classes]
    return dict(zip(available_classes_names, available_classes))


def create_class_from_module(name: str, args: dict, module):  # noqa: ANN001, ANN201, D103
    return get_class_from_module(name, module)(**args)


def get_class_from_module(name: str, module):  # noqa: ANN001, ANN201, D103
    available_classes = available_classes_in_module(module)
    raise_if_module_is_not_available(name, available_classes)

    return available_classes[name]


def create_torch_module(name: str, args: dict) -> torch.nn.Module:  # noqa: D103
    return create_class_from_module(name, args, torch.nn)


def get_torch_optimizer(name: str) -> torch.optim.Optimizer:  # noqa: D103
    return get_class_from_module(name, torch.optim)


def get_torch_lr_scheduler(name: str) -> torch.optim.lr_scheduler.LRScheduler:  # noqa: D103
    return get_class_from_module(name, torch.optim.lr_scheduler)


def raise_if_module_is_not_available(module_name: str, module_dict: dict[str, torch.nn.Module]) -> None:  # noqa: D103
    if module_name not in module_dict:
        raise ClassNotFoundInModuleError(module_name, list(module_dict))
