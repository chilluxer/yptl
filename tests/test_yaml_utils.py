import pytest
import yaml
from pytorch_lightning import Trainer

import yptl
from yptl.utilities.yaml import (
    create_model_from_yaml_config,
    create_trainer_from_config,
)


@pytest.fixture()
def yaml_config():
    yaml_string = """
Model:
    type: LightningMLP
    args:
      n_inp: 2
      n_out: 2
      hidden_layers: [32, 32]

Trainer:
    max_epochs: 1
    callbacks:
      - type: LearningRateMonitor
        args:
          logging_interval: step
"""

    return yaml.safe_load(yaml_string)


def test_create_model_from_yaml(yaml_config):
    model = create_model_from_yaml_config(yaml_config["Model"])

    assert isinstance(model, yptl.models.LightningMLP)


def test_create_trainer_from_yaml(yaml_config):
    trainer = create_trainer_from_config(yaml_config["Trainer"])
    assert isinstance(trainer, Trainer)
