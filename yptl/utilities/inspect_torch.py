from __future__ import annotations  # noqa: D100

import inspect

import torch.nn


class ClassNotFoundInModuleError(Exception):  # noqa: D101
    def __init__(self, module_name: str, available_modules: list) -> None:  # noqa: D107
        super().__init__(f"{module_name} not found in list of available modules: {' '.join(available_modules)}")


def available_modules_in_torch_nn() -> dict:  # noqa: D103
    available_module_classes = [cls for _, cls in inspect.getmembers(torch.nn, inspect.isclass)]
    available_module_names = [x.__name__ for x in available_module_classes]
    return dict(zip(available_module_names, available_module_classes))


def create_torch_module(name: str, args: dict) -> torch.nn.Module:  # noqa: D103
    available_modules = available_modules_in_torch_nn()
    raise_if_module_is_not_available(name, available_modules)

    return available_modules[name](**args)


def raise_if_module_is_not_available(module_name: str, module_dict: dict[str, torch.nn.Module]) -> None:  # noqa: D103
    if module_name not in module_dict:
        raise ClassNotFoundInModuleError(module_name, list(module_dict))
