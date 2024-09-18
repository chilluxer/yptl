from pytorch_lightning.callbacks import Callback


class MyPrintingCallback(Callback):
    """Ëxample callback taken from https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html"""
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")