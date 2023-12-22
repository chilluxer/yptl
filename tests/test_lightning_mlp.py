from __future__ import annotations

from argparse import Namespace

import pytest

from yptl.models import LightningMLP


@pytest.fixture()
def hparams():
    activation = {"type": "ReLU", "args": {}}
    return Namespace(n_inp=1, n_out=2, hidden_layers=[32, 32], activation=activation)


def test_create_model(hparams):
    LightningMLP(**vars(hparams))
