from __future__ import annotations  # noqa: D100

import inspect

import torch.nn
import torch.optim


class ClassNotFoundInModuleError(Exception):  # noqa: D101
    def __init__(self, module_name: str, available_modules: list) -> None:  # noqa: D107
        super().__init__(f"{module_name} not found in list of available modules: {' '.join(available_modules)}")


def available_classes_in_module(module) -> dict[str | type]:
    available_classes = [cls for _, cls in inspect.getmembers(module, inspect.isclass)]
    available_classes_names = [x.__name__ for x in available_classes]
    return dict(zip(available_classes_names, available_classes))


def create_class_from_module(name: str, args: dict, module):
    available_modules = available_classes_in_module(module)
    raise_if_module_is_not_available(name, available_modules)

    return available_modules[name](**args)


def create_torch_module(name: str, args: dict) -> torch.nn.Module:  # noqa: D103
    return create_class_from_module(name, args, torch.nn)


def create_torch_optimizer(name: str, args: dict) -> torch.optim.Optimizer:  # noqa: D103
    return create_class_from_module(name, args, torch.optim)


def create_torch_lr_scheduler(name: str, args: dict) -> torch.optim.Optimizer:  # noqa: D103
    return create_class_from_module(name, args, torch.optim.lr_scheduler)


def raise_if_module_is_not_available(module_name: str, module_dict: dict[str, torch.nn.Module]) -> None:  # noqa: D103
    if module_name not in module_dict:
        raise ClassNotFoundInModuleError(module_name, list(module_dict))
