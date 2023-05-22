from pytorch_lightning.callbacks import Callback
import torch
import torchmetrics
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

class LogParameters(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (batch_idx + 1) % trainer.log_every_n_steps == 0:
            for name, param in pl_module.named_parameters():
                trainer.logger.experiment.add_histogram(f"Model/{name}", param, global_step=trainer.global_step)
                trainer.logger.experiment.add_histogram(f"Model/{name}_grad", param.grad, global_step=trainer.global_step)


class LogConfusionMatrix(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        outputs = torch.cat([x['out'] for x in pl_module.validation_step_outputs], dim=0)
        labels = torch.cat([x['y'] for x in pl_module.validation_step_outputs], dim=0)
        confusion_matrix = torchmetrics.ConfusionMatrix(task = 'binary', num_classes=2, threshold=0.5).to(pl_module.device)
        confusion_matrix(outputs, labels.int())
        confusion_matrix_computed = confusion_matrix.compute().detach().cpu().numpy().astype(int)

        df_cm = pd.DataFrame(confusion_matrix_computed)
        plt.figure(figsize = (10,7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Blues').get_figure()
        plt.close(fig_)
        trainer.logger.experiment.add_figure("Confusion matrix", fig_, trainer.global_step)

class LogDistances(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        if 'distances' in pl_module.validation_step_outputs[0]:
            if trainer.current_epoch == 1:
                distances = torch.cat([x['distances'] for x in pl_module.validation_step_outputs], dim=0)
                trainer.logger.experiment.add_histogram(f"Distances/true", distances, global_step=trainer.global_step)
            predicted_distances = torch.cat([x['predicted_distances'] for x in pl_module.validation_step_outputs], dim=0)
            trainer.logger.experiment.add_histogram(f"Distances/predicted", predicted_distances, global_step=trainer.global_step)
