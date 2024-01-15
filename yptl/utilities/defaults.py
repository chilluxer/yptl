def add_default_optimizer(hparams) -> None:
    hparams["optimizer"] = {"type": "Adam", "args": {"lr": 1e-3}}


def add_default_loss_function(hparams) -> None:
    hparams["loss_fn"] = {"type": "MSELoss", "args": {}}
