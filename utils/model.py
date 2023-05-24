from pathlib import Path

import torch


def get_device():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    return device


def save_model(model, filename):
    """
    Saves model to disk.
    :param model: torch model
    :param filename: filename
    :return: None
    """
    if Path(filename).exists():
        raise FileExistsError(f"File {filename} already exists")

    torch.save(model.state_dict(), filename)


def load_model(filename):
    """
    Loads model from disk.
    :param filename: filename
    :return: state_dict of the model
    """
    if not Path(filename).exists():
        raise FileNotFoundError(f"File {filename} does not exist")

    state_dict = torch.load(filename)

    return state_dict


def save_checkpoint(filename, model, optimizer, config, edge_class_weights):
    """
    Saves the current state of the model.
    :param str filename: filename of the checkpoint. Stored in the model directory.
    :param torch.nn.Module model: model to save
    :param torch.optim.Optimizer optimizer: optimizer to save
    :param DotDict config: configuration of the model
    :param torch.Tensor edge_class_weights: edge class weights
    :return: None
    """
    torch.save({
        'config': config,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_weights': edge_class_weights
    }, filename)


def load_checkpoint(filename):
    """
    Loads the checkpoint from disk.
    :param str filename: filename of the checkpoint. Stored in the model directory.
    :return: (config, model_state_dict, optimizer_state_dict, class_weights)
    """
    return torch.load(filename)
