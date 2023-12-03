import inspect
from typing import Dict

import torch.nn


def available_modules_in_torch_nn() -> Dict:
    available_module_classes = [cls for _, cls in inspect.getmembers(torch.nn, inspect.isclass)]
    available_module_names = [x.__name__ for x in available_module_classes]
    available_modules = dict(zip(available_module_names, available_module_classes))

    return available_modules


def create_torch_module(name: str, args: Dict) -> torch.nn.Module:
    available_modules = available_modules_in_torch_nn()
    check_if_module_is_available(name, available_modules)

    return available_modules[name](**args)


def check_if_module_is_available(module_name: str, module_dict: Dict[str, torch.nn.Module]):
    if module_name not in module_dict.keys():
        raise ValueError(
            f"{module_name} not found in list of available modules: {' '.join(i for i in module_dict.keys())}",
        )
