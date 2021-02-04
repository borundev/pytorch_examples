import torch
import wandb
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
import numpy as np


def make_validation_epoch_end(neg_label=0, pos_label=1):
    """
    A function to make the class methods for validation_epoch_end and test_epoch_end

    :param labels: a dict with 'pos_label' and 'neg_label'
    :return:
    """
    def validation_epoch_end(self,outputs):
        ground_truths, predictions = zip(*outputs)
        predictions = torch.nn.Softmax(1)(torch.cat(predictions)).cpu().numpy()
        ground_truths = torch.cat(ground_truths).cpu().numpy().astype(np.int)
        aps_0 = average_precision_score(1-ground_truths, predictions[:, 0])
        aps_1 = average_precision_score(ground_truths, predictions[:, 1])
        auc_0 = roc_auc_score(1-ground_truths, predictions[:, 0])
        auc_1 = roc_auc_score(ground_truths, predictions[:, 1])

        self.log('val/avg_precision_{}'.format(neg_label),aps_0)
        self.log('val/avg_precision_{}'.format(pos_label), aps_1)
        self.log('val/roc_auc_{}'.format(neg_label), auc_0)
        self.log('val/roc_auc_{}'.format(pos_label), auc_1)

        fig = plt.figure(figsize=(20, 5))
        ax = fig.add_subplot(1, 1, 1)
        precision, recall, thresholds = precision_recall_curve(ground_truths, predictions[:, 1])

        ax.step(recall, precision, color='r', alpha=0.2,
                where='post',label=pos_label)
        ax.fill_between(recall, precision, step='post', alpha=0.2,
                        color='r')
        precision, recall, thresholds = precision_recall_curve(1 - ground_truths, predictions[:, 0])
        ax.step(recall, precision, color='b', alpha=0.2,
                where='post',label=neg_label)
        ax.fill_between(recall, precision, step='post', alpha=0.2,
                        color='b')

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_ylim([0.0, 1.05])
        ax.set_xlim([0.0, 1.05])
        ax.set_title('PR')
        ax.legend()
        plt.close()
        self.log('PR', wandb.Image(ax))

        fig = plt.figure(figsize=(20, 5))
        ax = fig.add_subplot(1, 1, 1)
        fpr, tpr, thresholds = roc_curve(ground_truths, predictions[:, 1])
        ax.step(fpr, tpr, color='r', alpha=0.2,
                where='post', label=pos_label)
        ax.fill_between(fpr, tpr, step='post', alpha=0.2,
                        color='r')
        fpr, tpr, thresholds = roc_curve(1 - ground_truths, predictions[:, 0])
        ax.step(fpr, tpr, color='b', alpha=0.2,
                where='post', label=neg_label)
        ax.fill_between(fpr, tpr, step='post', alpha=0.2,
                        color='b')

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_ylim([0.0, 1.05])
        ax.set_xlim([0.0, 1.05])
        ax.set_title('ROC')
        ax.legend()
        plt.close()
        self.log('ROC', wandb.Image(ax))
    return validation_epoch_end