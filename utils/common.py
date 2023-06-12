import pickle


class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def _n(v):
    return v.detach().cpu().numpy()


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
