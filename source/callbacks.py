from pytorch_lightning.callbacks import Callback

class LogParametersCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (batch_idx + 1) % trainer.log_every_n_steps == 0:
            for name, param in pl_module.named_parameters():
                trainer.logger.experiment.add_histogram(f"Model/{name}", param, global_step=trainer.global_step)
                trainer.logger.experiment.add_histogram(f"Model/{name}_grad", param.grad, global_step=trainer.global_step)