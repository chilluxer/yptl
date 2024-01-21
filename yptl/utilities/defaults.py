from typing import MutableMapping  # noqa: D100


def add_default_optimizer(hparams: MutableMapping) -> None:  # noqa: D103
    hparams["optimizer"] = {"type": "Adam", "args": {"lr": 1e-3}}


def add_default_loss_function(hparams: MutableMapping) -> None:  # noqa: D103
    hparams["loss_fn"] = {"type": "MSELoss", "args": {}}


def add_default_activation(hparams: MutableMapping) -> None:  # noqa: D103
    hparams["activation"] = {"type": "ReLU", "args": {}}
