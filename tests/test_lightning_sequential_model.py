import pytest
import torch

from yptl import LightningSequentialModel


@pytest.fixture()
def hparams():
    layer_1 = {"type": "Linear", "args": {"in_features": 1, "out_features": 2}}
    layer_2 = {"type": "ReLU", "args": {}}
    return {"model": [layer_1, layer_2]}


def test_create_model(hparams):
    model = LightningSequentialModel(**hparams)
    assert isinstance(model.model[0], torch.nn.Linear)
    assert isinstance(model.model[1], torch.nn.ReLU)


def test_forward_pass(hparams):
    model = LightningSequentialModel(**hparams)
    assert torch.allclose(
        model.forward(torch.zeros(1, 1)), torch.tensor([[0.00000000, 0.91861129]])
    )
