import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

from . import DotDict


def get_device():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    return device


def get_metrics(targets, predictions):
    """
    Computes the metrics for the given targets and predictions.
    :param targets: (batch_size, num_nodes, num_nodes)
    :param predictions: (batch_size, num_nodes, num_nodes)
    :return: DotDict with metrics
    """
    acc = accuracy_score(targets.flatten(), predictions.flatten())
    bal_acc = balanced_accuracy_score(targets.flatten(), predictions.flatten(), adjusted=True)
    precision, recall, f1_score, _ = precision_recall_fscore_support(targets.flatten(),
                                                                     predictions.flatten(),
                                                                     average='binary')
    return DotDict({
        "acc": acc,
        "bal_acc": bal_acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    })


def save_checkpoint(filename, model, optimizer, **kwargs):
    """
    Saves the current state of the model.
    :param str filename: filename of the checkpoint. Stored in the model directory.
    :param torch.nn.Module model: model to save
    :param torch.optim.Optimizer optimizer: optimizer to save
    :keyword DotDict config: configuration of the model
    :return: None
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        **kwargs
    }, filename)


def load_checkpoint(filename, device=torch.device('cpu')):
    """
    Loads the checkpoint from disk.
    :param str filename: filename of the checkpoint. Stored in the model directory.
    :param torch.device device: device to load the checkpoint to
    :return: (config, model_state_dict, optimizer_state_dict, class_weights)
    """
    return torch.load(filename, map_location=device)
