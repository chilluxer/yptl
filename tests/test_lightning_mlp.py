import pytest
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset

from yptl.models import LightningMLP
from yptl.utilities.inspect_torch import ClassNotFoundInModulesError


@pytest.fixture()
def hparams():
    activation = {"type": "ReLU", "args": {}}
    return {"n_inp": 2, "n_out": 2, "hidden_layers": [32, 32], "activation": activation}


@pytest.fixture()
def hparams_with_optimizer(hparams):
    hparams_with_opt = hparams
    hparams_with_opt["optimizer"] = {"type": "SGD", "args": {"lr": 0.001}}

    return hparams_with_opt


@pytest.fixture()
def dataloader():
    dataset = TensorDataset(torch.rand(50, 2), torch.rand(50, 2))
    return DataLoader(dataset, batch_size=4)


def test_create_model(hparams):
    LightningMLP(**hparams)


def test_run_forward_pass(hparams):
    model = LightningMLP(**hparams)
    assert torch.allclose(
        model.forward(torch.ones(1, 2)), torch.tensor([[0.14762419, 0.08727027]])
    )


def test_run_training_step(hparams, dataloader):
    model = LightningMLP(**hparams)
    trainer = Trainer(max_epochs=1, enable_checkpointing=False, logger=False)
    trainer.fit(model, dataloader)
    assert torch.allclose(
        trainer.logged_metrics["train_loss_epoch"], torch.tensor(0.29835150)
    )


def test_configure_optimizers_from_dict(hparams_with_optimizer):
    optim_dict = LightningMLP(**hparams_with_optimizer).configure_optimizers()
    assert isinstance(optim_dict["optimizer"], torch.optim.SGD)


def test_configure_default_optimizers(hparams):
    optim_dict = LightningMLP(**hparams).configure_optimizers()
    assert isinstance(optim_dict["optimizer"], torch.optim.Adam)


def test_raise_with_invalid_optimizer(hparams):
    with pytest.raises(ClassNotFoundInModulesError):
        LightningMLP(
            **hparams, optimizer={"type": "Banana", "args": {}}
        ).configure_optimizers()


def test_configure_lr_scheduler(hparams):
    optim_dict = LightningMLP(
        **hparams, lr_scheduler={"type": "ReduceLROnPlateau", "args": {"factor": 0.1}}
    ).configure_optimizers()
    assert isinstance(
        optim_dict["lr_scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau
    )


def test_load_from_checkpoint(hparams, dataloader, tmp_path):
    model = LightningMLP(
        **hparams, lr_scheduler={"type": "ReduceLROnPlateau", "args": {"factor": 0.1}}
    )

    from pytorch_lightning.callbacks import ModelCheckpoint

    checkpoint_callback = ModelCheckpoint(
        dirpath=tmp_path / "checkpoints/",
    )
    trainer = Trainer(max_epochs=1, callbacks=[checkpoint_callback], logger=False)
    trainer.test(model=model, dataloaders=dataloader)
    initial_loss = trainer.logged_metrics["test_loss"]
    trainer.fit(model, train_dataloaders=dataloader, val_dataloaders=dataloader)
    trainer.test(model=model, dataloaders=dataloader)
    loss_model = trainer.logged_metrics["test_loss"]
    assert loss_model < initial_loss
    # Force checkpoint to be loaded by using ckpt_path parameter
    # see: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html#lightning.pytorch.trainer.trainer.Trainer.test
    trainer.test(dataloaders=dataloader, ckpt_path="best")
    loss_ckpt = trainer.logged_metrics["test_loss"]

    assert torch.allclose(loss_model, loss_ckpt)
