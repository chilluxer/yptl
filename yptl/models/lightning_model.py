from __future__ import annotations  # noqa: D100

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import Namespace

import torch.nn
from pytorch_lightning import LightningModule

from yptl.utilities import create_torch_module


class LightningModel(LightningModule):  # noqa: D101
    def __init__(self: LightningModel, args: Namespace) -> None:  # noqa: D107
        super().__init__()
        self.save_hyperparameters()
        self.model = create_module_list_from_dictionary(args.model)

    def forward(self: LightningModel, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        y = x
        for layer in self.model:
            y = layer(y)
        return y


def create_module_list_from_dictionary(  # noqa: D103
    module_definitions: list[dict],
) -> torch.nn.ModuleList:
    module_list = torch.nn.ModuleList()
    for module_definition in module_definitions:
        name = module_definition["name"]
        arguments = module_definition["args"]
        module_list.append(create_torch_module(name, arguments))
    return module_list
