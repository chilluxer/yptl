import pytest
import torch

from yptl.utilities.inspect_torch import (
    ClassNotFoundInModulesError,
    create_torch_module,
    get_torch_lr_scheduler,
    get_torch_optimizer,
    load_sourcefile_as_module,
)

INVALID_MODULE_NAME = "banana"
INVALID_ARGUMENT = INVALID_MODULE_NAME


def test_create_torch_module():
    layer_dict = {"type": "Linear", "args": {"in_features": 2, "out_features": 5}}
    layer = create_torch_module(layer_dict["type"], layer_dict["args"])
    assert isinstance(layer, torch.nn.Linear)


def test_create_torch_module_throw_if_unknown_module_is_passed_in_dictionary():
    layer_dict = {"type": INVALID_MODULE_NAME, "args": {}}
    with pytest.raises(ClassNotFoundInModulesError):
        _ = create_torch_module(layer_dict["type"], layer_dict["args"])


def test_create_torch_module_catches_unknown_module_arguments():
    layer_dict = {"type": "Linear", "args": {INVALID_ARGUMENT: 4}}
    with pytest.raises(TypeError):
        _ = create_torch_module(layer_dict["type"], layer_dict["args"])


def test_get_torch_optimizer():
    optim_cls = get_torch_optimizer("Adam")
    assert optim_cls == torch.optim.Adam


def test_get_torch_lr_scheduler():
    lr_scheduler_cls = get_torch_lr_scheduler("ReduceLROnPlateau")
    assert lr_scheduler_cls == torch.optim.lr_scheduler.ReduceLROnPlateau


@pytest.mark.usefixtures("_change_cwd_to_test_dir")
def test_load_class_from_custom_source_file():
    load_sourcefile_as_module("./myclass.py")
    import myclass

    name = "MyClass"
    args = {"a": 5, "b": "Hello", "d": {"key": "value"}}
    instance = create_torch_module(name=name, args=args)
    assert isinstance(instance, myclass.MyClass)
    assert vars(instance) == args
