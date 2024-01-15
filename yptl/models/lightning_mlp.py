from __future__ import annotations  # noqa: D100

from typing import Optional, Union

import torch
from pytorch_lightning import LightningModule
from torch.nn import Linear

from yptl.utilities.defaults import add_default_loss_function, add_default_optimizer
from yptl.utilities.inspect_torch import create_torch_lr_scheduler, create_torch_module, create_torch_optimizer


class LightningMLP(LightningModule):  # noqa: D101
    activation_: torch.nn.Module
    model: torch.nn.Sequential

    def __init__(
        self,
        n_inp: int,
        n_out: int,
        hidden_layers: list[int],
        activation: dict[str, str | dict],
        optimizer: None | dict[str, str | dict] = None,
        loss_fn: None | dict[str, str | dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = torch.nn.Sequential()
        self.activation = create_torch_module(self.hparams.activation["type"], self.hparams.activation["args"])

        neurons = [self.hparams.n_inp, *self.hparams.hidden_layers, self.hparams.n_out]
        for i, _ in enumerate(neurons[:-2]):
            self.model.append(Linear(neurons[i], neurons[i + 1]))
            self.model.append(self.activation)
        self.model.append(Linear(neurons[-2], neurons[-1]))

        if not self.hparams.optimizer:
            add_default_optimizer(self.hparams)

        if not self.hparams.loss_fn:
            add_default_loss_function(self.hparams)

        self.loss_fn = create_torch_module(self.hparams.loss_fn["type"], self.hparams.loss_fn["args"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:  # noqa: ARG002, D102
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):  # noqa: ARG002, ANN201, D102
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch: torch.Tensor, batch_idx: int):  # noqa: ARG002, ANN201, D102
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("test_loss", loss, on_epoch=True)

    def configure_optimizers(self):  # noqa: ANN201, D102
        ret_dict = {}
        optimizer_dict = self.hparams.optimizer
        optimizer_dict["args"]["params"] = self.parameters()
        optimizer = create_torch_optimizer(optimizer_dict["type"], optimizer_dict["args"])
        ret_dict["optimizer"] = optimizer

        if "lr_scheduler" in self.hparams:
            scheduler_dict = self.hparams.lr_scheduler
            scheduler_dict["args"]["optimizer"] = optimizer
            lr_scheduler = create_torch_lr_scheduler(scheduler_dict["type"], scheduler_dict["args"])
            ret_dict["lr_scheduler"] = lr_scheduler
        return ret_dict
