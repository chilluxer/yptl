import pytest
from pytorch_lightning import seed_everything


@pytest.fixture(autouse=True)
def _set_global_seed():
    seed_everything(42)
