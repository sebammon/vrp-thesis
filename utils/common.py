import pickle

import torch


class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def _n(v):
    return v.detach().cpu().numpy()


def get_device():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    return device


def save_pickle(obj, file):
    """
    Stores the given object in a pickle file.
    :param obj: object to store
    :param file: location of pickle file
    :return: None
    """
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(file):
    """
    Loads the given pickle file.
    :param file: location of pickle file
    :return: Any
    """
    with open(file, "rb") as f:
        return pickle.load(f)


def load_config(**kwargs):
    """
    Loads the configuration for the model.
    :keyword int hidden_dim: hidden dimension of the model
    :keyword int node_features: number of node features
    :keyword int edge_distance_features: number of edge distance features
    :keyword int edge_types_features: number of edge type features
    :keyword int num_gcn_layers: number of GCN layers
    :keyword int num_mlp_layers: number of MLP layers
    :keyword int dropout: dropout probability
    :return: DotDict
    """
    config = DotDict({**kwargs})

    config.hidden_dim = kwargs.get("hidden_dim", 16)
    config.node_features = kwargs.get("node_features", 3)
    config.edge_distance_features = kwargs.get("edge_distance_features", 1)
    config.edge_types_features = kwargs.get("edge_types_features", 3)
    config.num_gcn_layers = kwargs.get("num_gcn_layers", 5)
    config.num_mlp_layers = kwargs.get("num_mlp_layers", 3)
    config.dropout = kwargs.get("dropout", None)

    return config


def save_checkpoint(filename, model, optimizer, **kwargs):
    """
    Saves the current state of the model.
    :param str filename: filename of the checkpoint. Stored in the model directory.
    :param torch.nn.Module model: model to save
    :param torch.optim.Optimizer optimizer: optimizer to save
    :keyword DotDict config: configuration of the model
    :return: None
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            **kwargs,
        },
        filename,
    )


def load_checkpoint(filename, device=torch.device("cpu")):
    """
    Loads the checkpoint from disk.
    :param str filename: filename of the checkpoint. Stored in the model directory.
    :param torch.device device: device to load the checkpoint to
    :return: (config, model_state_dict, optimizer_state_dict, class_weights)
    """
    return torch.load(filename, map_location=device)
