from pathlib import Path

import torch

MODELS_DIR = Path(__file__).parent.parent / 'models'


def get_device():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    return device


def save_model(model, filename='model.pt', overwrite=False):
    """
    Saves model to disk.
    :param model: torch model
    :param filename: filename
    :param overwrite: overwrite existing file
    :return: None
    """
    if not MODELS_DIR.exists():
        MODELS_DIR.mkdir()

    if (MODELS_DIR / filename).exists():
        if not overwrite:
            raise FileExistsError(f"File {filename} already exists")

    torch.save(model.state_dict(), MODELS_DIR / filename)


def load_model(filename):
    """
    Loads model from disk.
    :param filename: filename
    :return: torch model
    """
    if not MODELS_DIR.exists():
        raise FileNotFoundError(f"Directory {MODELS_DIR} does not exist")

    if not (MODELS_DIR / filename).exists():
        raise FileNotFoundError(f"File {filename} does not exist")

    model = torch.load(MODELS_DIR / filename)

    return model


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
    }, MODELS_DIR / filename)


def load_checkpoint(filename):
    """
    Loads the checkpoint from disk.
    :param str filename: filename of the checkpoint. Stored in the model directory.
    :return: (config, model_state_dict, optimizer_state_dict, class_weights)
    """
    return torch.load(MODELS_DIR / filename)
