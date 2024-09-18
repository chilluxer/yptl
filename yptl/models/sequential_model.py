from __future__ import annotations  # noqa: D100

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
from pytorch_lightning import LightningModule

from yptl.models.helper import (
    configure_optimizers_from_model_hparams,
    create_sequential_model,
)
from yptl.utilities.defaults import add_default_loss_function, add_default_optimizer
from yptl.dynamic_import import create_torch_module


class LightningSequentialModel(LightningModule):  # noqa: D101
    def __init__(  # noqa: D107
        self: LightningSequentialModel,
        model: list[dict[str, str | dict]],  # noqa: ARG002
        loss_fn: None | dict[str, str | dict] = None,  # noqa: ARG002
        optimizer: None | dict[str, str | dict] = None,  # noqa: ARG002
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = create_sequential_model(self.hparams.model)

        if not self.hparams.optimizer:
            add_default_optimizer(self.hparams)

        if not self.hparams.loss_fn:
            add_default_loss_function(self.hparams)

        self.loss_fn = create_torch_module(
            self.hparams.loss_fn["type"], self.hparams.loss_fn["args"]
        )

    def forward(self: LightningSequentialModel, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:  # noqa: ARG002, D102
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log(
            "train_loss",
            loss.detach(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
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
        return configure_optimizers_from_model_hparams(self)
