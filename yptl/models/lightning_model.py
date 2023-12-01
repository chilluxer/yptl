from argparse import Namespace
from typing import Dict, List

import torch.nn
from pytorch_lightning import LightningModule

from yptl.utilities import create_torch_module


class LightningModel(LightningModule):
    def __init__(self, args: Namespace):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_module_list_from_dictionary(args.model)

    def forward(self, input):
        x = input
        for layer in self.model:
            x = layer(x)
        return x


def create_module_list_from_dictionary(
    module_definitions: List[Dict],
) -> torch.nn.ModuleList:
    module_list = torch.nn.ModuleList()
    for module_definition in module_definitions:
        name = module_definition["name"]
        arguments = module_definition["args"]
        module_list.append(create_torch_module(name, arguments))
    return module_list
