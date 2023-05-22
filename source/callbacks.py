from pytorch_lightning.callbacks import Callback
import torch
import pandas as pnd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import metrics
import numpy as np

class LogParameters(Callback):
    """
    Logs histograms of model parameters and gradients to check for vanishing/exploding gradients
    """
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (batch_idx + 1) % trainer.log_every_n_steps == 0:
            for name, param in pl_module.named_parameters():
                trainer.logger.experiment.add_histogram(f"Model/{name}", param, global_step=trainer.global_step)
                trainer.logger.experiment.add_histogram(f"Model/{name}_grad", param.grad, global_step=trainer.global_step)

def get_metrics_and_curves(metric_type, y_pred, y_true, invert=False, threshold=0.5):
    """
    Calculate metrics and curves for a given metric type
    ROC: Receiver Operating Characteristic curve, metric = Area under the curve
    PR: Precision-Recall curve, metric = Area under the curve (Average precision)
    CM: Confusion Matrix, metric = F1 score

    Parameters
    ----------
    metric_type : str
        One of "ROC", "PR", "CM"
    y_pred : torch.Tensor
        Predicted labels
    y_true : torch.Tensor
        True labels
    invert : bool
        If True, do 1 - y_pred, use if y_pred is distance instead of probability

    Returns
    -------
    metric_value : float
        Value of the metric
    metric_disp : matplotlib.figure.Figure
        Figure of the curve/matrix
    """
    if invert:
        y_pred = 1 - y_pred
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    if metric_type == "ROC": 
        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred, pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)
        roc_disp = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
        return roc_auc, roc_disp.figure_
    elif metric_type == "PR":
        # Precision-Recall Curve
        precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred, pos_label=1)
        pr_auc = metrics.auc(recall, precision)
        pr_disp = metrics.PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=pr_auc).plot()
        return pr_auc, pr_disp.figure_
    elif metric_type == "CM":
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred > threshold)
        df_cm = pnd.DataFrame(confusion_matrix)
        plt.figure(figsize = (10,7))
        cm_disp = sns.heatmap(df_cm, annot=True, cmap='Blues').get_figure()
        plt.close(cm_disp)
        f1 = metrics.f1_score(y_true, y_pred > threshold)
        return f1, cm_disp



class LogMetrics(Callback):
    """
    Log metrics and curves for validation and training

    Scalars: ROC/val_AUC, ROC/train_AUC, PR/val_AUC, PR/train_AUC, CM/val_F1, CM/train_F1 
    Images: ROC/val, ROC/train, PR/val, PR/train, CM/val, CM/train
    """
    def on_validation_epoch_end(self, trainer, pl_module):
        outputs = torch.cat([x['out'] for x in pl_module.validation_step_outputs], dim=0)
        labels = torch.cat([x['y'] for x in pl_module.validation_step_outputs], dim=0)
        for metric, value in zip(["ROC", "PR", "CM"], ["AUC", "AUC", "F1"]):
            metric_value, metric_disp = get_metrics_and_curves(metric, outputs, labels)
            pl_module.log(f"{metric}/val_{value}", metric_value)
            if trainer.current_epoch % 10 == 0:
                trainer.logger.experiment.add_figure(f"{metric}/val", metric_disp, global_step=trainer.global_step)

    def on_train_epoch_end(self, trainer, pl_module):
        outputs = torch.cat([x['out'] for x in pl_module.train_step_outputs], dim=0)
        labels = torch.cat([x['y'] for x in pl_module.train_step_outputs], dim=0)
        for metric, value in zip(["ROC", "PR", "CM"], ["AUC", "AUC", "F1"]):
            metric_value, metric_disp = get_metrics_and_curves(metric, outputs, labels)
            pl_module.log(f"{metric}/train_{value}", metric_value)
            if trainer.current_epoch % 10 == 0:
                trainer.logger.experiment.add_figure(f"{metric}/train", metric_disp, global_step=trainer.global_step)

class LogDistances(Callback):
    """
    Logs histograms of true and predicted distances and their difference
    """
    def on_validation_epoch_end(self, trainer, pl_module):
        if 'distances' in pl_module.validation_step_outputs[0]:
            distances = torch.cat([x['distances'] for x in pl_module.validation_step_outputs], dim=0).cpu().detach().numpy()
            if trainer.current_epoch == 1:
                trainer.logger.experiment.add_histogram(f"Distances/true", 
                                                        distances, 
                                                        global_step=trainer.global_step)
            predicted_distances = torch.cat([x['predicted_distances'] for x in pl_module.validation_step_outputs], dim=0).cpu().detach().numpy()
            trainer.logger.experiment.add_histogram(f"Distances/predicted", 
                                                    predicted_distances, 
                                                    global_step=trainer.global_step)
            trainer.logger.experiment.add_histogram(f"Distances/difference", 
                                                    np.abs(predicted_distances - distances), 
                                                    global_step=trainer.global_step)

class ClearOutputs(Callback):
    """
    Clears the outputs of the model after each epoch
    """
    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.train_step_outputs.clear()
        
    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        pl_module.validation_step_outputs.clear()
