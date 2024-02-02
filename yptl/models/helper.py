from __future__ import annotations  # noqa: D100

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pytorch_lightning import LightningModule
    from pytorch_lightning.utilities.types import OptimizerLRScheduler

from yptl.utilities.inspect_torch import create_torch_module, get_torch_lr_scheduler, get_torch_optimizer


def configure_optimizers_from_model_hparams(model: LightningModule) -> OptimizerLRScheduler:  # noqa: D103
    ret_dict = {}
    optimizer_dict = model.hparams.optimizer
    optimizer_cls = get_torch_optimizer(optimizer_dict["type"])
    optimizer = optimizer_cls(params=model.parameters(), **optimizer_dict["args"])
    ret_dict["optimizer"] = optimizer

    if "lr_scheduler" in model.hparams:
        scheduler_dict = model.hparams.lr_scheduler
        lr_scheduler_cls = get_torch_lr_scheduler(scheduler_dict["type"])
        lr_scheduler = lr_scheduler_cls(optimizer, **scheduler_dict["args"])
        ret_dict["lr_scheduler"] = lr_scheduler

        # ReduceLROnPleateau requires an additional "monitor" value
        # see_ https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        if lr_scheduler_cls is torch.optim.lr_scheduler.ReduceLROnPlateau:
            ret_dict["monitor"] = "val_loss"

    return ret_dict


def create_sequential_model(  # noqa: D103
    layer_definitions: list[dict],
) -> torch.nn.ModuleList:
    model = torch.nn.Sequential()
    for module_definition in layer_definitions:
        name = module_definition["type"]
        arguments = module_definition["args"]
        model.append(create_torch_module(name, arguments))
    return model
