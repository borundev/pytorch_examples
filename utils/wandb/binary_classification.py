import sklearn
import torch
import wandb
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

        predictions = predictions[:,1]
        pred = (predictions > 0.5).astype(np.int)

        aps_0 = sklearn.metrics.average_precision_score(1-ground_truths, 1-predictions)
        aps_1 = sklearn.metrics.average_precision_score(ground_truths, predictions)
        auc_0 = sklearn.metrics.roc_auc_score(1-ground_truths, 1-predictions)
        auc_1 = sklearn.metrics.roc_auc_score(ground_truths, predictions)

        self.log('val/avg_precision_{}'.format(neg_label),aps_0)
        self.log('val/avg_precision_{}'.format(pos_label), aps_1)
        self.log('val/roc_auc_{}'.format(neg_label), auc_0)
        self.log('val/roc_auc_{}'.format(pos_label), auc_1)

        ax = confusion_matrix(ground_truths, pred, pos_label, neg_label)
        self.log('CM', wandb.Image(ax))

        ax = pr_curve(ground_truths,predictions,pos_label,neg_label)
        self.log('PR', wandb.Image(ax))

        ax = roc_curve(ground_truths, predictions, pos_label, neg_label)
        self.log('ROC', wandb.Image(ax))
    return validation_epoch_end


def roc_curve(y_true, y_score, pos_label,neg_label, figsize=(20,5)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score)
    ax.step(fpr, tpr, color='r', alpha=0.2,
            where='post', label=pos_label)
    ax.fill_between(fpr, tpr, step='post', alpha=0.2,
                    color='r')
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(1 - y_true, 1 - y_score)
    ax.step(fpr, tpr, color='b', alpha=0.2,
            where='post', label=neg_label)
    ax.fill_between(fpr, tpr, step='post', alpha=0.2,
                    color='b')

    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.05])
    ax.set_title('ROC')
    ax.legend()
    plt.close()
    return ax

def pr_curve(y_true, y_score, pos_label, neg_label, figsize=(20, 5)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true, y_score)

    ax.step(recall, precision, color='r', alpha=0.2,
            where='post', label=pos_label)
    ax.fill_between(recall, precision, step='post', alpha=0.2,
                    color='r')
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(1 - y_true, 1 - y_score)
    ax.step(recall, precision, color='b', alpha=0.2,
            where='post', label=neg_label)
    ax.fill_between(recall, precision, step='post', alpha=0.2,
                    color='b')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.05])
    ax.set_title('PR')
    ax.legend()
    plt.close()
    return ax

def confusion_matrix(y_true,pred,pos_label, neg_label, figsize=(10, 10)):
    cm = sklearn.metrics.confusion_matrix(y_true, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=figsize)
    ax.matshow(cm, cmap=plt.cm.Blues)
    for i in range(2):
        for j in range(2):
            c = cm[j, i]
            ax.text(i, j, str(c), va='center', ha='center', color='Red', size=20)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([neg_label, pos_label])
    ax.set_yticks([0, 1])
    ax.set_yticklabels([neg_label, pos_label])
    plt.close()
    return ax