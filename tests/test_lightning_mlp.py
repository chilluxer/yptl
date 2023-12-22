import pytest
import torch
from pytorch_lightning import seed_everything

from yptl.models import LightningMLP


@pytest.fixture()
def hparams():
    activation = {"type": "ReLU", "args": {}}
    return {"n_inp": 2, "n_out": 2, "hidden_layers": [32, 32], "activation": activation}


def test_create_model(hparams):
    LightningMLP(**hparams)


def test_run_forward_pass(hparams):
    model = LightningMLP(**hparams)
    assert torch.allclose(model.forward(torch.ones(1, 2)), torch.tensor([[0.14762419, 0.08727027]]))
