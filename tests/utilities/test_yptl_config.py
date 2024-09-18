from __future__ import annotations

import pytest
import yaml

from yptl.yptl_config import YPTLConfig, add_args_keyword_to_factory_dicts


@pytest.fixture()
def yaml_input():
    yaml_string = """
datamodule:
    type: MNISTDataModule
model:
    type: LightningMLP
    args:
      n_inp: 2
      n_out: 2
      hidden_layers: [32, 32]

trainer:
    max_epochs: 1
    callbacks:
      - type: LearningRateMonitor
        args:
          logging_interval: step
      - type: LearningRateMonitor
"""
    return yaml.safe_load(yaml_string)


def test_input_will_be_lowercased(tmp_path, yaml_input):
    fname = tmp_path / "config.yaml"
    capitalized = {key.upper(): value for key, value in yaml_input.items()}

    with open(fname, "w") as f:
        yaml.dump(capitalized, f)

    _ = YPTLConfig.from_yaml(fname)


def test_add_args_keyword_to_factory_dicts(yaml_input):
    add_args_keyword_to_factory_dicts(yaml_input)
    assert isinstance(yaml_input["datamodule"]["args"], dict)
    assert all(
        [
            isinstance(callback["args"], dict)
            for callback in yaml_input["trainer"]["callbacks"]
        ]
    )


def test_add_args_keyword_to_factory_dicts2(yaml_input):
    config = YPTLConfig(**yaml_input)
    assert isinstance(config.datamodule["args"], dict)
    assert all(
        [isinstance(callback["args"], dict) for callback in config.trainer["callbacks"]]
    )
