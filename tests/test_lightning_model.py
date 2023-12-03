from argparse import Namespace

import pytest
import torch

from yptl import LightningModel

INVALID_MODULE_NAME = "banana"
INVALID_ARGUMENT = INVALID_MODULE_NAME
MANUAL_SEED = 42


@pytest.fixture()
def hparams():
    layer_1 = {"name": "Linear", "args": {"in_features": 1, "out_features": 2}}
    layer_2 = {"name": "ReLU", "args": {}}
    return Namespace(model=[layer_1, layer_2])


def test_forward_pass_of_model_in_LightningModel(hparams):
    torch.manual_seed(MANUAL_SEED)
    model = LightningModel(hparams)
    model_output = model.forward(torch.zeros(1, 1))
    torch.manual_seed(MANUAL_SEED)
    layer_output = torch.nn.ReLU()(torch.nn.Linear(1, 2)(torch.zeros(1, 1)))
    assert torch.allclose(model_output, layer_output)


def test_throw_if_unknown_module_is_passed_in_dictionary():
    layer_1 = {"name": INVALID_MODULE_NAME, "args": {}}
    hparams = Namespace(model=[layer_1])
    with pytest.raises(ValueError):
        LightningModel(hparams)


def test_create_model_from_module_dict(hparams):
    model = LightningModel(hparams)
    assert isinstance(model.model[0], torch.nn.Linear)
    assert isinstance(model.model[1], torch.nn.ReLU)


def test_torch_catches_unknown_module_arguments(hparams):
    hparams.model[0]["args"][INVALID_ARGUMENT] = 4
    with pytest.raises(TypeError):
        LightningModel(hparams)
