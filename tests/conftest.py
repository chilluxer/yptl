import pytest
import torch
from pytorch_lightning import seed_everything


@pytest.fixture(autouse=True)
def _set_global_seed():
    seed_everything(42)
    torch.set_printoptions(precision=8)


@pytest.fixture(autouse=True)
def _set_torch_precision():
    torch.set_printoptions(precision=8)
