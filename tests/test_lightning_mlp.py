from argparse import Namespace
from typing import List

import pytest
import torch.nn
from pytorch_lightning import LightningModule
from torch.nn import Linear

from yptl.utilities import create_torch_module


@pytest.fixture
def hparams():
    activation = {"type": "ReLU", "args": {}}
    return Namespace(n_inp=1, n_out=2, hidden_layers=[32, 32], activation=activation)


class LightningMLP(LightningModule):
    activation_: torch.nn.Module
    n_inp_: int
    n_out_: int
    hidden_layers_: List[int]
    model: torch.nn.Sequential

    def __init__(self, hparams: Namespace):
        super().__init__()
        self.n_inp_ = hparams.n_inp
        self.n_out_ = hparams.n_out
        self.hidden_layers_ = hparams.hidden_layers

        self.model = torch.nn.Sequential()
        self.activation = create_torch_module(hparams.activation["type"], hparams.activation["args"])

        neurons = [self.n_inp_] + self.hidden_layers_ + [self.n_out_]
        for i, _ in enumerate(neurons[:-1]):
            self.model.append(Linear(neurons[i], neurons[i]))
            self.model.append(self.activation)
        self.model.append(Linear(neurons[-2], neurons[-1]))

    def forward(self, X: torch.tensor):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass


def test_create_model(hparams):
    LightningMLP(hparams)
