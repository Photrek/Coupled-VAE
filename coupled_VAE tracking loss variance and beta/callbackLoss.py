from pytorch_lightning.callbacks import Callback

class LossLoggerCallback(Callback):
    def __init__(self, config_name):
        super().__init__()
        self.config_name = config_name

        # Create the file immediately when the callback is initialized
        with open(self.config_name, "w") as f:
            f.write(f"Epoch\tLoss\n")

    def on_train_epoch_end(self, trainer, pl_module):
        # Access the 'loss' from the logged metrics
        loss = trainer.callback_metrics.get('loss', None)
        if loss is not None:
            with open(self.config_name, "a") as f:
                f.write(f"{trainer.current_epoch}\t{loss.item()}\n")