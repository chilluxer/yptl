from __future__ import annotations  # noqa: D100

from typing import Union

import torch
from pytorch_lightning import LightningModule
from torch.nn import Linear

from yptl.utilities import create_torch_module


class LightningMLP(LightningModule):  # noqa: D101
    activation_: torch.nn.Module
    model: torch.nn.Sequential

    def __init__(self, n_inp: int, n_out: int, hidden_layers: list[int], activation: dict[str, Union[str, dict]]):  # noqa: ANN204, ARG002, D107, UP007
        super().__init__()
        self.save_hyperparameters()

        self.model = torch.nn.Sequential()
        self.activation = create_torch_module(self.hparams.activation["type"], self.hparams.activation["args"])

        neurons = [self.hparams.n_inp, *self.hparams.hidden_layers, self.hparams.n_out]
        for i, _ in enumerate(neurons[:-1]):
            self.model.append(Linear(neurons[i], neurons[i]))
            self.model.append(self.activation)
        self.model.append(Linear(neurons[-2], neurons[-1]))

    def forward(self, x: torch.tensor) -> torch.Tensor:  # noqa: D102
        pass

    def training_step(self, batch, batch_idx) -> None:  # noqa: ANN001, D102
        pass

    def test_step(self, batch, batch_idx):  # noqa: ANN001, ANN201, D102
        pass

    def validation_step(self, batch, batch_idx):  # noqa: ANN001, ANN201, D102
        pass

    def configure_optimizers(self):  # noqa: ANN201, D102
        pass
